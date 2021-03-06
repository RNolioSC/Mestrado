# --------------------------------
#     PDU DESCRIPTION
# --------------------------------
# forma de organizacao do pacote, criar um novo
PDU_DEFAULT = {
    'attribute': ['pdu'],
    'size': [1],
    'payload': [],
    'length': 1
}

PDU_ANA = {
    'attribute': ['endereco', 'umidade', 'temperatura', 'gas', 'chamas'],
    'size': [2, 2, 2, 2, 1],
    'payload': ['endereco', 'umidade', 'temperatura', 'gas', 'chamas'],
    'length': 9
}

PDU_RFID = {
    'attribute': ['tag'],
    'size': [8],
    'payload': ['tag'],
    'length': 9
}

PDU_TYPE = {
    b'\x11': PDU_ANA,
    b'\x12': PDU_RFID
}
