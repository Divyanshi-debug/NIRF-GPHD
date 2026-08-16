import csv, pathlib
p = pathlib.Path('gphd_2023.csv')
out = pathlib.Path('gphd_2023_transformed.csv')
with p.open(newline='', encoding='utf-8') as f:
    reader = list(csv.reader(f))
if not reader:
    raise SystemExit('Empty input')
header = reader[0]
col_index = {c: i for i,c in enumerate(header)}
req = ['ft_year1','ft_year2','ft_year3','serial_number','GPHD']
for r in req:
    if r not in col_index:
        raise SystemExit('Missing column: %s' % r)
rows = []
for row in reader[1:]:
    def safe_float(val):
        try:
            return float(val)
        except Exception:
            return 0.0
    ft1 = safe_float(row[col_index['ft_year1']])
    ft2 = safe_float(row[col_index['ft_year2']])
    ft3 = safe_float(row[col_index['ft_year3']])
    total_ft = ft1 + ft2 + ft3
    nphd = total_ft / 3.0
    gphd = safe_float(row[col_index['GPHD']])
    f_gphd = gphd / 20.0
    serial = row[col_index['serial_number']]
    rows.append((serial, total_ft, nphd, gphd, f_gphd))
rows.sort(key=lambda x: x[3], reverse=True)
with out.open('w', newline='', encoding='utf-8') as f:
    w = csv.writer(f)
    w.writerow(['serial_number','total_ft','nphd','GPHD','f(gphd)'])
    for r in rows:
        w.writerow([r[0], ('%.6f' % r[1]).rstrip('0').rstrip('.'), ('%.6f' % r[2]).rstrip('0').rstrip('.'), ('%.6f' % r[3]).rstrip('0').rstrip('.'), ('%.6f' % r[4]).rstrip('0').rstrip('.')])
print('WROTE', len(rows), 'rows to', out)
print('Top 6 rows:')
for r in rows[:6]:
    print(r)
