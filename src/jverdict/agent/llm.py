"""Claude API 호출 래퍼.

- 구조화 출력(output_config.format = JSON 스키마)과 strict 툴로 역할별 결과를 받는다. 강제 tool_choice는 쓰지 않는다
  (claude-sonnet-5-5는 거부). 응답 블록은 고치지 않고 그대로 대화에 붙인다(사고 블록 보존 규칙).
- 호출마다 토큰과 비용을 기록하고, 요청·응답을 캐시에 남긴다. 재실행(replay) 모드는 API를 부르지 않고 캐시를 순서대로 돌려준다.
  요청 지문이 다르면(입력이 바뀌었으면) 멈춘다. 지문은 두 가지다: 원본 그대로(key)와, 요청 속 긴 소수를 유효숫자
  10자리로 맞춘 정규화 지문(key_norm). 같은 코드·데이터라도 부동소수 합의 마지막 비트가 실행 환경에 따라
  달라질 수 있어서, 재실행은 둘 중 하나가 같으면 같은 입력으로 본다.
- 비밀값: 키는 config.get_secret으로만 받고, 오류 메시지에서 키 모양 문자열과 실제 키 값을 지운다.
- Claude에 보내는 것은 집계 통계와 스키마뿐이다(원자료 행은 보내지 않음). 툴 결과도 집계 요약만 돌려준다.
"""
from __future__ import annotations

import hashlib
import json
import re
import time
from dataclasses import asdict, dataclass, field
from typing import Callable

from jverdict.config import get_secret

MODEL = "claude-sonnet-5-5"
EFFORT = "medium"
# 달러 / 100만 토큰 (입력, 출력, 캐시 읽기, 캐시 쓰기 5분)
PRICES = {"claude-sonnet-5-5": (2.00, 10.00, 0.20, 2.50), "claude-haiku-5-5": (0.10, 0.50, 0.01, 0.125)}

_KEY_PAT = re.compile(r"sk-ant-[A-Za-z0-9_\-]{6,}")
_LONG_FLOAT = re.compile(r"-?\d+\.\d{8,}(?:[eE][-+]?\d+)?")
_URL_KEY = re.compile(r"([?&](?:key|apikey|api_key|serviceKey|projectKey)=)[^&\s]+", re.I)


def sanitize(text: str) -> str:
    """오류·로그 문자열에서 비밀값을 지운다."""
    s = _KEY_PAT.sub("[비밀값 제거]", str(text))
    s = _URL_KEY.sub(r"\1[비밀값 제거]", s)
    for name in ("ANTHROPIC_API_KEY", "JEJU_HUB_PROJECT_KEY"):
        v = get_secret(name)
        if v and len(v) >= 8:
            s = s.replace(v, "[비밀값 제거]")
    return s


class LLMError(RuntimeError):
    pass


class ReplayMismatch(LLMError):
    pass


@dataclass
class CallRecord:
    role: str
    key: str
    key_norm: str
    model: str
    response: dict
    usage: dict
    cost_usd: float
    seconds: float


@dataclass
class LLM:
    mode: str = "claude"                    # claude / replay
    model: str = MODEL
    effort: str = EFFORT
    cache: list[dict] = field(default_factory=list)     # replay용 이전 기록
    records: list[CallRecord] = field(default_factory=list)
    max_tokens: int = 32000
    _i: int = 0
    _client: object = None
    # 재실행 중 '정확한 지문'으로 맞은 옛 기록(key_norm 없음)의 정규화 지문: {기록 순번: key_norm}
    # 입력이 비트 단위로 같았으므로 지금 계산한 key_norm이 곧 원래 입력의 key_norm이다(orchestrator.backfill_key_norm)
    norm_backfill: dict = field(default_factory=dict)

    # ------------------------------------------------------------------ 기반
    @property
    def available(self) -> bool:
        return self.mode == "replay" or bool(get_secret("ANTHROPIC_API_KEY"))

    def _client_obj(self):
        if self._client is None:
            import anthropic
            key = get_secret("ANTHROPIC_API_KEY")
            if not key:
                raise LLMError("ANTHROPIC_API_KEY가 설정되지 않음(.env 또는 환경변수)")
            self._client = anthropic.Anthropic(api_key=key, max_retries=2, timeout=180)
        return self._client

    @staticmethod
    def _key(payload: dict) -> str:
        return hashlib.sha256(json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str).encode()).hexdigest()

    @staticmethod
    def _key_norm(payload: dict) -> str:
        """긴 소수(소수점 아래 8자리 이상)를 유효숫자 10자리로 맞춘 뒤의 지문: 마지막 비트 차이에 흔들리지 않는다."""
        s = json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str)
        s = _LONG_FLOAT.sub(lambda m: format(float(m.group()), ".10g"), s)
        return hashlib.sha256(s.encode()).hexdigest()

    def _cost(self, model: str, u: dict) -> float:
        pin, pout, pcr, pcw = PRICES.get(model, PRICES[MODEL])
        return (u.get("input_tokens", 0) * pin + u.get("output_tokens", 0) * pout
                + (u.get("cache_read_input_tokens") or 0) * pcr + (u.get("cache_creation_input_tokens") or 0) * pcw) / 1e6

    def _create(self, role: str, system: str, messages: list, tools: list | None, schema: dict) -> dict:
        payload = {"model": self.model, "effort": self.effort, "system": system, "messages": messages,
                   "tools": tools or [], "schema": schema}
        key, key_norm = self._key(payload), self._key_norm(payload)
        t0 = time.time()
        if self.mode == "replay":
            if self._i >= len(self.cache):
                raise ReplayMismatch(f"재실행 기록에 {role} 호출이 더 없음")
            rec = self.cache[self._i]
            self._i += 1
            same = rec.get("key") == key or (rec.get("key_norm") is not None and rec["key_norm"] == key_norm)
            if rec["role"] != role or not same:
                raise ReplayMismatch(f"{role} 호출의 입력이 원래 실행과 다름(데이터·코드·설계가 바뀌었는지 확인)")
            if rec.get("key") == key and rec.get("key_norm") is None:
                self.norm_backfill[self._i - 1] = key_norm
            resp = rec["response"]
        else:
            import anthropic
            kw = dict(model=self.model, max_tokens=self.max_tokens, system=system, messages=messages,
                      output_config={"effort": self.effort, "format": {"type": "json_schema", "schema": schema}},
                      cache_control={"type": "ephemeral"})
            if tools:
                kw["tools"] = tools
            try:
                # 긴 사고·출력도 끊기지 않게 스트리밍으로 받고, 완성된 메시지만 쓴다
                with self._client_obj().messages.stream(**kw) as stream:
                    resp = stream.get_final_message().to_dict()
            except anthropic.APIStatusError as e:
                raise LLMError(sanitize(f"Claude API 오류 {e.status_code}: {getattr(e, 'message', e)}")) from None
            except anthropic.APIConnectionError:
                raise LLMError("Claude API에 연결할 수 없음(인터넷 확인). 템플릿 모드로 계속할 수 있음") from None
            except anthropic.APIError as e:
                raise LLMError(sanitize(f"Claude API 오류: {e}")) from None
        u = resp.get("usage", {}) or {}
        self.records.append(CallRecord(role, key, key_norm, self.model, resp, u, self._cost(self.model, u),
                                       round(time.time() - t0, 2)))
        return resp

    @staticmethod
    def _final_json(resp: dict) -> dict:
        sr = resp.get("stop_reason")
        if sr == "refusal":
            raise LLMError("모델이 응답을 거절함")
        if sr == "max_tokens":
            raise LLMError("응답이 길이 상한에 걸려 잘림")
        texts = [b["text"] for b in resp.get("content", []) if b.get("type") == "text"]
        if not texts:
            raise LLMError("텍스트 응답이 없음")
        try:
            return json.loads(texts[-1])
        except json.JSONDecodeError as e:
            raise LLMError(f"JSON 해석 실패: {e}") from None

    # ------------------------------------------------------------------ 공개 함수
    def json_call(self, role: str, system: str, user: str, schema: dict) -> dict:
        return self._final_json(self._create(role, system, [{"role": "user", "content": user}], None, schema))

    def agent_call(self, role: str, system: str, user: str, schema: dict, tools: list[dict],
                   run_tool: Callable[[str, dict], tuple[str, bool]], max_turns: int = 10,
                   on_tool: Callable[[str, dict, str, bool], None] | None = None) -> dict:
        """툴 루프. 모델이 툴을 다 쓰고 끝내면 마지막 JSON을 돌려준다."""
        messages: list = [{"role": "user", "content": user}]
        for _ in range(max_turns):
            resp = self._create(role, system, messages, tools, schema)
            content = resp.get("content", [])
            messages.append({"role": "assistant", "content": content})
            sr = resp.get("stop_reason")
            if sr == "tool_use":
                results = []
                for b in content:
                    if b.get("type") != "tool_use":
                        continue
                    out, is_err = run_tool(b["name"], b.get("input") or {})
                    if on_tool:
                        on_tool(b["name"], b.get("input") or {}, out, is_err)
                    results.append({"type": "tool_result", "tool_use_id": b["id"], "content": out,
                                    **({"is_error": True} if is_err else {})})
                messages.append({"role": "user", "content": results})
                continue
            if sr == "pause_turn":
                continue
            return self._final_json(resp)
        raise LLMError(f"{role}: 툴 호출 {max_turns}회 안에 끝나지 않음")

    # ------------------------------------------------------------------ 기록
    def summary(self) -> dict:
        tot = {"calls": len(self.records), "input_tokens": 0, "output_tokens": 0, "cache_read_input_tokens": 0,
               "cache_creation_input_tokens": 0, "cost_usd": 0.0, "by_role": {}}
        for r in self.records:
            for k in ("input_tokens", "output_tokens", "cache_read_input_tokens", "cache_creation_input_tokens"):
                tot[k] += int(r.usage.get(k) or 0)
            tot["cost_usd"] += r.cost_usd
            br = tot["by_role"].setdefault(r.role, {"calls": 0, "cost_usd": 0.0})
            br["calls"] += 1
            br["cost_usd"] += r.cost_usd
        tot["cost_usd"] = round(tot["cost_usd"], 5)
        for v in tot["by_role"].values():
            v["cost_usd"] = round(v["cost_usd"], 5)
        tot["model"], tot["effort"], tot["mode"] = self.model, self.effort, self.mode
        return tot

    def cache_dump(self) -> list[dict]:
        return [{"role": r.role, "key": r.key, "key_norm": r.key_norm, "model": r.model, "response": r.response, "usage": r.usage,
                 "cost_usd": r.cost_usd} for r in self.records]
