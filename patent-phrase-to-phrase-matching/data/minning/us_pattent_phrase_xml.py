from xml.etree import ElementTree as ET
from xml.etree.ElementTree import Element
import re
import os
import pandas as pd


from utils.functions import KaggleFunctions as KF

# Iterar sobre cada clave, y asignar los valores en otro diccionario anidado, despues se aplanara
def get_data_from_xml_with(xml_mapper, xml_root_tag, plain_xml_data=None, plain_key=""):
    if plain_xml_data is None:
        plain_xml_data = {}

    xml_root_tag = dict([('__elem__', xml_root_tag), (xml_root_tag.tag, list(xml_root_tag))])

    tag_key = xml_mapper['__key__'] if '__key__' in xml_mapper else ""
    tag_value = xml_mapper['__value__'] if '__value__' in xml_mapper else ""
    data_name = xml_mapper['__name__'] if '__name__' in xml_mapper else ""

    if tag_key in xml_root_tag:
        if data_name not in plain_xml_data:
            plain_xml_data[data_name] = {}
        plain_key = xml_root_tag['__elem__'].text
        plain_xml_data[data_name][xml_root_tag['__elem__'].text] = []

    if tag_value in xml_root_tag:
        plain_xml_data[data_name][plain_key] += [xml_root_tag['__elem__'].text]

    for tag, elems in xml_root_tag.items():
        if tag in xml_mapper:
            for elem in elems:
                plain_xml_data, plain_key = get_data_from_xml_with(xml_mapper[tag], elem, plain_xml_data=plain_xml_data, plain_key=plain_key)

    return plain_xml_data, plain_key


if __name__ == '__main__':
    # schemes_path = os.path.join(os.path.dirname(__file__), '..', '..', 'resources', 'CPCSchemeXML202105')
    # cpc_scheme_A = ET.parse(os.path.join(schemes_path, 'cpc-scheme-A.xml'))
    # cpc_scheme_A_root = cpc_scheme_A.getroot()
    #
    # elem: Element = list(cpc_scheme_A_root)[0]
    # dict([(cpc_scheme_A_root.tag, list(cpc_scheme_A_root))])

    # KF.combine_dicts({'a': {'b': [1,3,4]}}, {'a': {'b': [5]}})

    cpc_scheme_xml_map = {
        'class-scheme': {
            'classification-item': {
                '__name__': 'section',
                '__key__': 'classification-symbol',
                'class-title': {
                    'title-part' : {
                        '__name__': 'section',
                        '__value__': 'text'
                    }
                },
                'classification-item': {
                    '__name__': 'class',
                    '__key__': 'classification-symbol',
                    'class-title': {
                        'title-part' : {
                            '__name__': 'class',
                            '__value__': 'text'
                        }
                    },
                    'classification-item' : {
                        '__name__': 'class',
                        '__key__': 'classification-symbol',
                        'class-title': {
                            'title-part' : {
                                '__name__': 'class',
                                '__value__': 'text'
                                # reference.text reference.CPC-specific-text.text
                            }
                        },
                        'classification-item' : {
                            '__name__': 'subclass',
                            '__key__': 'classification-symbol',
                            'class-title': {
                                'title-part' : {
                                    '__name__': 'subclass',
                                    '__value__': 'text'
                                    # reference.text reference.CPC-specific-text.text
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    cpc_data = {}
    cpc_data_merged = {}
    cpc_path = "/kaggle/input/cpc-scheme-cml-202105/CPCSchemeXML202105/"
    cpc_path = "./CPCSchemeXML202105/"

    resources_path = os.path.join(os.path.dirname(__file__), '..', '..', 'resources')
    cpc_path = os.path.join(resources_path, 'CPCSchemeXML202105')


    dirname, _, filenames = list(os.walk(cpc_path))[0]
    for filename in filenames:
        if re.match(r"cpc-scheme-[a-z]\.xml", filename, re.IGNORECASE):
            print(filename)
            cpc_scheme = ET.parse(os.path.join(dirname, filename))
            cpc_scheme_root = cpc_scheme.getroot()
            section = re.sub(r"cpc-scheme-([a-z])\.xml", r"\1", filename, flags=re.IGNORECASE)
            cpc_data[section], _ = get_data_from_xml_with(cpc_scheme_xml_map, cpc_scheme_root)
            # cpc_data_merged = combine_dicts(cpc_data_merged, cpc_data[section])

    cpc_data_merged = KF.combine_dicts(*list(cpc_data.values()))

    print(cpc_data_merged.keys())
    print(cpc_data_merged['section'])
    print({k: cpc_data_merged['section'][k] for k in sorted(cpc_data_merged['section'])})

    from collections import defaultdict
    cpc_list = []

    for name in cpc_data_merged:
        # Se ordena
        cpc_data_merged[name] = {k: cpc_data_merged[name][k] for k in sorted(cpc_data_merged[name])}

    # Al ser un conjuntos agregados es decir una subclass tiene una class y una section
    # Se puede rellenar la subclass y todos los atributos de class y subclass
    for code, subclass_values in cpc_data_merged['subclass'].items():
        _cpc_dict = defaultdict(tuple)

        _section = code[0]
        _class = code[1:-1]
        _subclass = code[-1]

        _cpc_dict["sect_class"] = _section+_class
        _cpc_dict["section"] = _section
        _cpc_dict["class"] = _class
        _cpc_dict["subclass"] = _subclass

        # Se rellena Section
        _aux_section_values = tuple([])
        for v in cpc_data_merged['section'][_section]:
            _aux_section_values = _aux_section_values + tuple([v])

        # Se rellena Class
        _aux_class_values = tuple([])
        for v in cpc_data_merged['class'][f'{_section}{_class}']:
            _aux_class_values = _aux_class_values + tuple([v])

        # Se rellena Subclass
        _aux_subclass_values = tuple([])
        for v in subclass_values:
            _aux_subclass_values = _aux_subclass_values + tuple([v])


        _cpc_dict[f'section_text'] = _aux_section_values
        _cpc_dict[f'class_text'] = _aux_class_values
        _cpc_dict[f'subclass_text'] = _aux_subclass_values

        cpc_list.append(_cpc_dict)

    cpc_df = pd.DataFrame(cpc_list, columns=['sect_class', 'section', 'class', 'subclass', 'section_text', 'class_text', 'subclass_text'])

    print(cpc_df.iloc[0:9])
    print(cpc_df.describe())

    cpc_df.to_csv(os.path.join(resources_path, "cpc_202105.csv"), index=False)

    cpc_df = pd.read_csv(os.path.join(resources_path, "cpc_202105.csv"))
    cpc_df['class'] = cpc_df['sect_class'].apply(lambda x: x[1:])

    print(cpc_df.describe())