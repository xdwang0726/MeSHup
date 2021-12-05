#!/bin/bash

# remove unrelated files from the download PMC XML file using pmc filter list
cd /home/xdwang/scratch/PMC/pmc_xml_2021/1

# Exit if the directory isn't found.
if (($?>0)); then
    echo "Can't find work dir... exiting"
    exit
fi

for i in *; do
    if ! grep -qxFe "$i" /home/xdwang/scratch/PMC/pmc_xml_2021/pmc00_30.txt; then
        echo "Deleting: $i"
        # the next line is commented out.  Test it.  Then uncomment to removed the files
        rm "$i"
    fi
done

# split the filted PMC file into small folers
cd /home/xdwang/scratch/PMC/pmc_xml/35-40/

n=0
for i in *
do
  if [ $((n+=1)) -gt 5 ]; then
    n=1
  fi
  todir=../35-40-$n
  [ -d "$todir" ] || mkdir "$todir"
  mv "$i" "$todir"
done

