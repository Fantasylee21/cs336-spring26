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

Problem (train_bpe_tinystories)
(a)

train on Apple M5 MacbookPro, 10 cores, 24GB RAM

Training time: 124.1 seconds (2.1 minutes)

Peak memory: 14871.9 MB

Longest token ID: 7160

Longest token length: 15 bytes

Longest token (UTF-8): 'Ġaccomplishment'

make sense

(b) 每次合并都要全量扫描 pair_counts 找最大值

Problem (train_bpe_expts_owt):  BPE Training on OpenWebText 
trained on Intel(R) Xeon(R) Processor @ 2.90GHz 16 cores 400G
Training time: 43417.6 seconds (723.6 minutes)
Peak memory: 101518.2 MB
Vocabulary size: 32000
Number of merges: 31743
Longest token ID: 25822
Longest token length: 64 bytes
Longest token (UTF-8): 'ÃÂÃÂÃÂÃÂÃÂÃÂÃÂÃÂÃÂÃÂÃÂÃÂÃÂÃÂÃÂÃÂ'

Problem (tokenizer_experiments):  Experiments with tokenizers

TinyStories Tokenizer :
macbook:
TinyStories: 10751 bytes / 2663 tokens = 4.04 bytes/token
throughput: 3230.36 total_bytes/sec
OpenWebText: 50460 bytes / 14821 tokens = 3.40 bytes/token
throughput: 2947.30 total_bytes/sec
alibaba server:
[TinyStories]
  10,751 bytes -> 2,663 tokens (4.04 bytes/token)
  4.05s, throughput: 2,654.32 bytes/s

[OpenWebText]
  50,460 bytes -> 14,821 tokens (3.40 bytes/token)
  20.36s, throughput: 2,478.30 bytes/s

OpenWebText Tokenizer :
[TinyStories]
  10,751 bytes -> 2,760 tokens (3.90 bytes/token)
  14.42s, throughput: 745.32 bytes/s

[OpenWebText]
  50,460 bytes -> 11,201 tokens (4.50 bytes/token)
  60.11s, throughput: 839.51 bytes/s

825GB = 825 * 1024 * 1024 * 1024 = 885,837,004,800 bytes

TinyStories Tokenizer :
time1 = 885,837,004,800 / 2947.30 = 300,500,000 seconds = 3.48 days
time2 = 885,837,004,800 / 3230.36 = 274,000,000 seconds = 3.17 days

OpenWebText Tokenizer :
time1 = 885,837,004,800 / 839.51 = 1,060,000,000 seconds = 11.5 days
time1 = 885,837,004,800 / 745.32 = 1,500,000,000 seconds = 16.5 days

(d) uint16 的取值范围是 0–65535，而 10K vocab 的 token ID 最大只到 9999