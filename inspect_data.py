import openpyxl

wb = openpyxl.load_workbook("data/commodity_prices.xlsx", data_only=True)
sheet = wb["Monthly Prices"]

# Print the first 6 rows so we can see the header structure
for row in sheet.iter_rows(min_row=1, max_row=6, values_only=True):
    print(row)
    print("---")