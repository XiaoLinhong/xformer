set -e # stop the shell on first error

# 12时起报
# BegTime=$(date -d"2024-05-31" '+%s')
# BegTime=$(date -d"2024-06-30" '+%s')
# EndTime=$(date -d"2024-07-31" '+%s')

 BegTime=$(date -d"2024-12-31" '+%s')
 EndTime=$(date -d"2025-01-31" '+%s')

# 执行
while [ $BegTime -le $EndTime ]; do
    thisTime=$(date -d@${BegTime} +%Y%m%d00)
    echo $thisTime
    python preprocessing.py $thisTime
    python forecast.py $thisTime
    let BegTime=BegTime+24*3600
done

