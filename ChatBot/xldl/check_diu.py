import re

text = "Diu 1. Säp xép các do'n vi hanh chinh cäp tinh"

re_dieu_1 = re.compile(r'^\s*(?:Điều|Diu|Diều|Biều|Ðiều)\s+(\d+)\.[\s\t]*(.*)$', re.IGNORECASE)
re_dieu_2 = re.compile(r'^\s*(?:Điều|D[iI][uU]?|Diều|Biều|Ðiều)\s+(\d+)\bs*[\.\:]?[\s\t]*(.*)$', re.IGNORECASE)

print("Original text:", repr(text))
print("Match 1:", re_dieu_1.match(text))
print("Match 2:", re_dieu_2.match(text))

# Let's inspect each char
for i, c in enumerate(text[:10]):
    print(f"Char {i}: {repr(c)} (ord: {ord(c)})")
