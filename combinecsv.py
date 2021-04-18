import glob

interesting_files = glob.glob("./CSVs/combined/*.csv") 
print(interesting_files)
header_saved = False
with open('generatedCombined.csv','w') as fout:
    for filename in interesting_files:
        with open(filename) as fin:
            header = next(fin)
            print(header)
            if not header_saved:
                fout.write(header)
                header_saved = True
            for line in fin:
                fout.write(line)