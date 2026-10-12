import gzip
import zlib
try:
    import bz2
except:
    bz2 = None
try:
    import lzma
except:
    lzma = None

pb_path = r"C:\Users\user\.gemini\antigravity\conversations\2decef6b-c9e1-4fba-bb8a-2e336dd93461.pb"

with open(pb_path, "rb") as f:
    header = f.read(64)
    f.seek(0)
    data = f.read()

print("Header (hex):", header.hex())

# Try GZIP
try:
    decomp = gzip.decompress(data)
    print("Successfully decompressed using gzip!")
    data = decomp
except Exception as e:
    print(f"Gzip failed: {e}")

# Try Zlib/Deflate
try:
    decomp = zlib.decompress(data)
    print("Successfully decompressed using zlib!")
    data = decomp
except Exception as e:
    print(f"Zlib failed: {e}")

# Try raw deflate
try:
    decomp = zlib.decompress(data, -zlib.MAX_WBITS)
    print("Successfully decompressed using raw deflate!")
    data = decomp
except Exception as e:
    print(f"Raw deflate failed: {e}")

# If compressed, let's search decompressed
targets = [b"Almgren", b"ExecutionCost", b"optimal_trajectory"]
for target in targets:
    pos = data.find(target)
    if pos != -1:
        print(f"Found '{target.decode()}' in potentially decompressed data at {pos}!")
    else:
        print(f"'{target.decode()}' not found.")
