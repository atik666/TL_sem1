import glob
import piexif

nfiles = 0
for filename in glob.iglob('/home/admin1/Documents/Atik/Imagenet/partitioned/A/train/**/*.JPEG', recursive=True):
#    try:
    nfiles = nfiles + 1
    print("About to process file %d, which is %s." % (nfiles,filename))
    piexif.remove(filename)
        
#    except Exception as e:
#        print (e.message)
#        print(e.args)
#        pass