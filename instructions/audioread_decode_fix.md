# Fix for `audioread` decode error

OS: Amazon Linux 2018.03 | 64-bit (x86) Amazon Machine Image (AMI)


## Problem:

`audioread.NoBackendError`

==> Missing `ffmpeg` - need to build from source.


## Solution:

```
sudo su -
cd /usr/local/bin
mkdir ffmpeg
cd ffmpeg

wget https://www.johnvansickle.com/ffmpeg/old-releases/ffmpeg-3.4.2-64bit-static.tar.xz
tar xf ffmpeg-3.4.2-64bit-static.tar.xz
mv ffmpeg-3.4.2-64bit-static/* .
exit

sudo ln -s /usr/local/bin/ffmpeg/ffmpeg /usr/bin/ffmpeg
```

## Check installation:

Then run `ffmpeg` to check if it's been installed successfully. You should get output like this:

```
ffmpeg version 4.1.3-static https://johnvansickle.com/ffmpeg/  Copyright (c) 2000-2019 the FFmpeg developers
  built with gcc 6.3.0 (Debian 6.3.0-18+deb9u1) 20170516
  configuration: --enable-gpl --enable-version3 --enable-static --disable-debug --disable-ffplay --disable-indev=sndio --disable-outdev=sndio --cc=gcc-6 --enable-fontconfig --enable-frei0r --enable-gnutls --enable-gmp --enable-gray --enable-libaom --enable-libfribidi --enable-libass --enable-libvmaf --enable-libfreetype --enable-libmp3lame --enable-libopencore-amrnb --enable-libopencore-amrwb --enable-libopenjpeg --enable-librubberband --enable-libsoxr --enable-libspeex --enable-libvorbis --enable-libopus --enable-libtheora --enable-libvidstab --enable-libvo-amrwbenc --enable-libvpx --enable-libwebp --enable-libx264 --enable-libx265 --enable-libxml2 --enable-libxvid --enable-libzvbi --enable-libzimg
  libavutil      56. 22.100 / 56. 22.100
  libavcodec     58. 35.100 / 58. 35.100
  libavformat    58. 20.100 / 58. 20.100
  libavdevice    58.  5.100 / 58.  5.100
  libavfilter     7. 40.101 /  7. 40.101
  libswscale      5.  3.100 /  5.  3.100
  libswresample   3.  3.100 /  3.  3.100
  libpostproc    55.  3.100 / 55.  3.100
Hyper fast Audio and Video encoder
usage: ffmpeg [options] [[infile options] -i infile]... {[outfile options] outfile}...

```
