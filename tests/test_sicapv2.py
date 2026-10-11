import io
import zipfile

import numpy as np
import pytest

from modern_pca import sicapv2 as s


def test_metadata_parser():
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, 'w') as archive:
        archive.writestr('xl/worksheets/sheet1.xml', '''<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main"><sheetData>
        <row><c r="A1" t="inlineStr"><is><t>name</t></is></c><c r="B1" t="inlineStr"><is><t>value</t></is></c></row>
        <row><c r="A2" t="inlineStr"><is><t>patient1</t></is></c><c r="B2"><v>3</v></c></row>
        </sheetData></worksheet>''')
    assert s.table(buffer.getvalue(), ['name', 'value']) == [{'name': 'patient1', 'value': 3}]
    with pytest.raises(ValueError):
        s.table(buffer.getvalue(), ['wrong'])


@pytest.mark.parametrize('grade', range(4))
def test_official_binary_mapping(grade):
    keys = ('NC', 'G3', 'G4', 'G5')
    row = dict(zip(keys, [int(i == grade) for i in range(4)]))
    row['G4C'] = 1  # Ancillary cribriform annotation is not a fifth class.
    assert s.primary_label(row) == (keys[grade], int(grade != 0))


@pytest.mark.parametrize('values', [[0,0,0,0], [1,1,0,0], [0,0,0,2]])
def test_ambiguous_labels_rejected(values):
    with pytest.raises(ValueError):
        s.primary_label(dict(zip(('NC','G3','G4','G5'), values)))


def test_patient_exclusivity():
    assert s.patient_partition([{'patient_id':'a'}], [{'patient_id':'b'}])['overlap'] == 0
    with pytest.raises(ValueError):
        s.patient_partition([{'patient_id':'a'}], [{'patient_id':'a'}])


def test_mask_interpretation():
    result = s.mask_summary(np.array([[0,1],[2,3]], dtype=np.uint8))
    assert result['mask_value_0']==result['mask_value_1']==result['mask_value_2']==result['mask_value_3']==1
    assert result['mixed_zero_nonzero'] and result['multiple_nonzero_values']
    assert s.mask_summary(np.array([[0,3,4,5]],dtype=np.uint8))['majority_nonzero_value']==3
    with pytest.raises(ValueError):
        s.mask_summary(np.array([[[4]]], dtype=np.uint8))


def test_duplicates_and_order_coverage():
    rows = [dict(sample_id=str(i), file_sha256=str(i // 2)) for i in range(7)]
    assert s.duplicate_groups(rows) == [['0','1'], ['2','3'], ['4','5']]
    assert s.sample_order(rows) == s.sample_order(list(reversed(rows)))
    assert sorted(r['sample_id'] for r in s.sample_order(rows)) == [str(i) for i in range(7)]


@pytest.mark.parametrize('path', ['/tmp/x','SICAPv2/../x','SICAPv2\\x','VAL2/x'])
def test_archive_escape_rejected(path):
    with pytest.raises(ValueError):
        s.safe_member(path)
