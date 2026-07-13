Problem (unicode1):  Understanding Unicode

(a) '\x00'

(b) Printed representation such as 'print' is human-readable, while __repr__() is meant to be unambiguous and can be used to recreate the object if needed.

(c) 
```
>>> chr(0)
'\x00'
>>> print(chr(0))

>>> "this is a test" + chr(0) + "string"
'this is a test\x00string'
>>> print("this is a test" + chr(0) + "string")
this is a teststring
```

Problem (unicode2):  Unicode Encodings

(a) UTF-8 byte tokenizers yield far smaller, more compact representations for common Latin, numeric, and ASCII text than UTF-16/UTF-32, which waste space with fixed two/four-byte codepoints for basic characters

(b) This test with ASCII "hello" coincidentally works, but passing multi-byte text like "中文".encode("utf-8") will throw UnicodeDecodeError. The encoding rule of UTF-8: A single Unicode character is encoded as 1 to 4 consecutive bytes. Not every byte corresponds to a single character independently.

```
def decode_utf8_bytes_to_str_wrong(bytestring: bytes) -> str:
    return bytestring.decode("utf-8")

# Test call
res = decode_utf8_bytes_to_str_wrong("hello呵呵".encode("utf-8"))
print(res)

```

(c) The first byte 0xFF starts with eight 1 bits, which UTF-8 defines as an invalid leading byte with no valid character length encoding, making this two-byte sequence entirely undecodable to any Unicode character under the UTF-8 standard.