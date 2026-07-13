def decode_utf8_bytes_to_str_wrong(bytestring: bytes) -> str:
    return bytestring.decode("utf-8")

# Test call
res = decode_utf8_bytes_to_str_wrong("hello呵呵".encode("utf-8"))
print(res)
