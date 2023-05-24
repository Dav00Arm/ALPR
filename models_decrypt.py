import hashlib
from Crypto.Cipher import AES


def load_graph(path):
    with open(path, 'rb') as f:
        protobuf_byte_str = f.read()
    return protobuf_byte_str


def decrypt_file(enc, _key):
    key = hashlib.sha256(_key.encode()).digest()
    iv = enc[:AES.block_size]
    cipher = AES.new(key, AES.MODE_CBC, iv)
    s = cipher.decrypt(enc[AES.block_size:])
    return s[:-ord(s[len(s) - 1:])]



