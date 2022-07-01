for f in **/*/*-.dump; do 
    mv -- "$f" "${f%-.dump}.dump"
done


for f in **/*/*-.json; do 
    mv -- "$f" "${f%-.json}.json"
done
