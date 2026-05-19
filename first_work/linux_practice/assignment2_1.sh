mkdir -p linux_practice/docs
mkdir -p linux_practice/backup
touch linux_practice/docs/readme.txt
touch linux_practice/docs/notes.log
touch linux_practice/docs/temp.tmp
rm linux_practice/docs/temp.tmp
mv linux_practice/docs/notes.log linux_practice/docs/daily_report.txt
echo "Project Status:Active" > linux_practice/docs/daily_report.txt
echo "Current Date:$(date)" >> linux_practice/docs/daily_report.txt
cp linux_practice/docs/*.txt linux_practice/backup/
chmod 644 linux_practice/docs/readme.txt
chmod 644 linux_practice/docs/daily_report.txt
echo "Archive Complete.File readme.txt is now read-only."
echo "Archive Complete.File daily_report.txt is now read-only."

