#!/bin/bash

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