#26 617.4 Traceback (most recent call last):
#26 617.4   File "./tools/codegen.py", line 360, in <module>
#26 617.4     main()
#26 617.4   File "./tools/codegen.py", line 322, in main
#26 617.4     src = SourceFile( filename )
#26 617.4   File "./tools/codegen.py", line 171, in __init__
#26 617.4     self._text = fd.read()
#26 617.4   File "/opt/conda/lib/python3.6/encodings/ascii.py", line 26, in decode
#26 617.4     return codecs.ascii_decode(input, self.errors)[0]
#26 617.4 UnicodeDecodeError: 'ascii' codec can't decode byte 0xe2 in position 21255: ordinal not in range(128)