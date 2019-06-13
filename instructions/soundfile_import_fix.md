# Fix compatability problem of SoundFile python package

OS: Amazon Linux 2018.03 | 64-bit (x86) Amazon Machine Image (AMI)

## Install SoundFile Package

`pip install SoundFile`

==> This will lead to this error: `OSError: library not found: 'sndfile'`

## Solution:

(1) Download the missing file
```
wget http://www.mega-nerd.com/libsndfile/files/libsndfile-1.0.28.tar.gz
tar xf libsndfile-1.0.28.tar.gz
```

(2) cd into the folder:
```
./configure --prefix=/usr    \
            --disable-static \
            --docdir=/usr/share/doc/libsndfile-1.0.28 &&
make
```

(3) Then as the **ROOT USER** (`sudo su`):
`make install`

(4) Then run again, you would probably see this error: `OSError: cannot load library 'libsndfile.so.1': libsndfile.so.1: cannot open shared object file: No such file or directory`

+ `ldconfig -p | grep libsndfile*` ==> you will get a path like this: `/usr/share/doc/libsndfile-1.0.28`  
+ `LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/share/doc/libsndfile-1.0.28`  
+ `export LD_LIBRARY_PATH`  
+ `sudo ldconfig`  

Now it's all set.
