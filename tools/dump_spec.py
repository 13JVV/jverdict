import sys, openpyxl
wb = openpyxl.load_workbook(sys.argv[1], read_only=True, data_only=True)
want = sys.argv[2].split(',') if len(sys.argv) > 2 else None
for ws in wb.worksheets:
    if want and not any(ws.title.startswith(w) for w in want):
        continue
    print(f"\n##### {ws.title}")
    for row in ws.iter_rows(values_only=True):
        cells = [str(c).strip().replace('\n', ' ') for c in row if c is not None and str(c).strip()]
        if cells:
            print(' | '.join(cells)[:400])
