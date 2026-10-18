"""Inspect only the explicitly named SICAPv2 archive; no training or VAL2 access."""
from collections import Counter
import json
import io
from pathlib import Path
import zipfile
from xml.etree import ElementTree as ET

NS={'s':'http://schemas.openxmlformats.org/spreadsheetml/2006/main'}


def read_xlsx(payload):
    with zipfile.ZipFile(io.BytesIO(payload)) as book:
        shared=[]
        if 'xl/sharedStrings.xml' in book.namelist():
            shared=[''.join(node.itertext()) for node in ET.fromstring(book.read('xl/sharedStrings.xml')).findall('s:si',NS)]
        result={}
        for name in sorted(n for n in book.namelist() if n.startswith('xl/worksheets/sheet') and n.endswith('.xml')):
            rows=[]
            for row in ET.fromstring(book.read(name)).findall('.//s:sheetData/s:row',NS):
                values=[]
                for cell in row.findall('s:c',NS):
                    column=0
                    for char in ''.join(c for c in cell.attrib['r'] if c.isalpha()): column=column*26+ord(char)-64
                    while len(values)<column: values.append(None)
                    value=cell.find('s:v',NS)
                    if cell.get('t')=='s': value=shared[int(value.text)]
                    elif cell.get('t')=='inlineStr': value=''.join(cell.find('s:is',NS).itertext())
                    elif cell.get('t')=='e': raise ValueError('Spreadsheet contains error cell')
                    elif value is not None:
                        value=value.text
                        if cell.get('t') not in ('str','d'):
                            number=float(value); value=int(number) if number.is_integer() else number
                    values[column-1]=value
                rows.append(values)
            result[name]=rows
        return result

ROOT=Path('/mnt/d/DSML 2025/Semester 1 (2026)/DSA5206/Project/data/sicapv2_v2')


def main():
    with zipfile.ZipFile(ROOT/'SICAPv2.zip') as archive:
        files=[info for info in archive.infolist() if not info.is_dir()]
        print(json.dumps(dict(files=len(files),bytes=sum(info.file_size for info in files),
            suffixes=Counter(Path(info.filename).suffix for info in files),
            directories=Counter(str(Path(info.filename).parent) for info in files)),indent=2))
        for info in files:
            if Path(info.filename).suffix.lower() not in ('.jpg','.jpeg','.png','.tif','.tiff'):
                print(info.filename,info.file_size)
                if Path(info.filename).suffix.lower()=='.txt':
                    print(archive.read(info).decode('utf-8',errors='replace'))
                if Path(info.filename).suffix.lower()=='.xlsx':
                    for sheet,rows in read_xlsx(archive.read(info)).items():
                        print(json.dumps(dict(sheet=sheet,rows=len(rows),first_rows=rows[:5])))
        print('First entries:',[info.filename for info in files[:12]])


if __name__=='__main__': main()
