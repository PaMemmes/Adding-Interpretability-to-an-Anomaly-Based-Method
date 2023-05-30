#!/bin/bash +x
cp -r git_repos/theZoo/malware/Binaries/ .
mv Binaries theZoo
cd theZoo
# Unzip every package in folder

# Unzip every package in folder
7z x -o* .

# Delete unnecessary files
find . -type f -iname \*.pass -delete
find . -type f -iname \*.zip -delete
find . -type f -iname \*.sha -delete
find . -type f -iname \*.md5 -delete
find . -type f -iname \*.sha256 -delete
find . -type f -iname \*.sha1 -delete
find . -type f -iname \*.shasum -delete
find . -type f -iname \*.txt -delete
find . -type f -iname \*.jpg -delete

# Delete specific viruses
rm -r \{*
rm -r AndroRat_6Dec2013
rm -r cryptowall/
rm -r EquationGroup*
rm -r OSX.*
rm -r Raccoon.Stealer.v2
rm -r SpyEye/
rm -r SymbOS.Lasco/
rm -r Trojan.AlienSpy/
rm -r Trojan.Asprox/
rm -r VBS.NewLove.A/
rm -r W32.Beagle/
rm -r W32.Mytob_EJ/
rm -r W32.NetSky/
rm -r Win32.GravityRat/
rm -r Win32.Turla.V1/
rm -r WM.Minimal.AB/
rm -r WM.NJ_WMVCK2_T/
rm -r ZeroLocker/

# Sanitize spaces in filenames and directories
find . -name "* *" -type d | rename 's/ /_/g'
find . -name "* *" -type f | rename 's/ /_/g'

# Split the files with the following script
mkdir -p  /home/pascal/splits/split
for d in */; do
        mkdir /home/pascal/splits/split/"$d"
done

cp /home/pascal/bash_files/Dockerfile /home/pascal/splits
for d in */; do
        for f in "$d"/*; do
                split -l 10 -d -a 5 "$f" /home/pascal/splits/split/"$f"
        done
done
