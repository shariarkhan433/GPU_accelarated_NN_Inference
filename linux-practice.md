# Linux commands practice lab

Everything runs inside ~/lab. Assumes Ubuntu/Debian with sudo. Each block has a task and a check. Try to predict the output before you run it.

## 0. Setup

```bash
mkdir -p ~/lab && cd ~/lab
pwd
clear
date
uname -a; uptime; whoami
```

Check: pwd prints /home/<you>/lab. Run `exit` last in the session, not now.

## 1. Directories: pwd, mkdir, rmdir, ls, cd

```bash
mkdir project old a/b/c        # fails: a does not exist
mkdir -p a/b/c                 # works
ls -l; ls -la; ls -R a
cd a/b/c && pwd
cd ../.. && pwd
cd -                           # back to previous dir
cd ~/lab
rmdir old                      # works, empty
rmdir a                        # fails, not empty
rmdir -p a/b/c                 # removes the whole empty chain
```

Check: explain why `rmdir a` fails and `rmdir -p a/b/c` does not.

## 2. Creating files: all nine methods from the sheet

```bash
touch f1.txt
touch f2.txt f3.txt f4.txt
> g1.txt
> g2.txt > g3.txt
cat >> h1.txt                  # type two lines, then Ctrl+D
echo >> e1.txt
echo >> e2.txt >> e3.txt       # which of e2/e3 gets the newline?
printf "" >> p1.txt
printf "line one\n line two\n line three\n" >> p2.txt
nano n1.txt                    # type something, Ctrl+O, Enter, Ctrl+X
vim v1.txt                     # i, type, Esc, :wq
gedit x1.txt                   # GUI, skip on a headless box
emacs -nw m1.txt               # Ctrl+X Ctrl+S, Ctrl+X Ctrl+C
ls -l
```

Check: `ls -l` shows sizes. Both e2 and e3 exist, but only the last redirection receives the output. Verify with `wc -c e2.txt e3.txt`.

## 3. Content commands: cat, head, tail, tac, more, less, sed, nano

```bash
seq 1 25 > nums.txt
cat nums.txt
cat > note.txt                 # type text, Ctrl+D
cat nums.txt note.txt > both.txt
head nums.txt
head -n 3 nums.txt
tail nums.txt
tail -n 3 nums.txt
tail -f /var/log/syslog        # Ctrl+C to stop (or journalctl -f)
tac nums.txt | head -3
more nums.txt                  # Space, Enter, b, /12, q
less nums.txt                  # /12, n, N, G, g, q
sed -n '1,4p' nums.txt
sed -n '20,$p' nums.txt
sed 's/1/ONE/' nums.txt | head -5       # print only
sed -i.bak '3d' nums.txt                 # edit in place, keeps nums.txt.bak
diff nums.txt nums.txt.bak
nano notes.txt
```

Check: after the sed -i, `wc -l nums.txt` is 24 and the .bak has 25.

## 4. Pipes and filters: cat, tac, grep

```bash
cat nums.txt | tac | cat | cat | tac | head -3    # same as plain head
printf "celena-90\nravi-72\nmina-95\njoy-48\n" > marks.txt
cat marks.txt | grep 9
grep -n 9 marks.txt
grep -v 9 marks.txt
grep -c 9 marks.txt
grep -i CELENA marks.txt
grep -E '9[0-9]$' marks.txt
ls -l /etc | grep conf
ps aux | grep -i bash
```

Check: why does `grep 9` match mina-95 and celena-90 but also a line like joy-49? Run it and see.

## 5. Copy, move, remove, rename: rm, cp, mv, rename

```bash
mkdir src dst
cp f1.txt f1_copy.txt
cp f1.txt f2.txt f3.txt dst/
cp -R src dst/src_copy
cp -a src dst/src_archive      # preserves perms, timestamps, links
mv f4.txt dst/
mv dst/f4.txt f4_renamed.txt   # move out and rename in one go
rm f1_copy.txt
rm -i f2.txt                   # interactive
rm -r dst/src_copy
touch a.txt b.txt c.txt
rename 's/\.txt$/.pdf/' a.txt b.txt c.txt   # Debian's perl rename
ls *.pdf
rename 's/\.pdf$/.txt/' *.pdf
```

Check: if `rename` is the util-linux version on your distro, the syntax differs. `rename --version` tells you which one you have.

## 6. Searching: find, locate

```bash
find . -name "*.txt"
find . -type f -name "*.txt" -size -1k
find . -type d
find . -mmin -10                           # modified in last 10 minutes
find . -name "*.txt" -exec wc -l {} +
find /etc -name "sysctl.conf" 2>/dev/null
sudo apt install -y plocate || sudo apt install -y mlocate
sudo updatedb
locate sysctl.conf
locate -i "*.conf" | head
```

Check: touch a new file, run `locate` for it (not found), then `sudo updatedb` and run it again (found). That is the database-staleness point from the sheet. `find` finds it immediately.

## 7. Permissions: chmod, chown, umask

```bash
touch perm.txt; ls -l perm.txt
chmod 325 perm.txt; ls -l perm.txt         # -wx-w-r-x
chmod 755 perm.txt
chmod 644 perm.txt
chmod u+x,g-w,o=r perm.txt
chmod +x perm.txt
chmod -w perm.txt; echo hi >> perm.txt     # permission denied
chmod 777 perm.txt
stat -c '%a %A %U:%G %n' perm.txt
sudo chown root perm.txt
sudo chown $USER:$USER perm.txt
umask; umask 022; touch u22.txt
umask 077; touch u77.txt; umask 022
ls -l u22.txt u77.txt
```

Drill: write each in octal without looking: rwxr-x---, r--r--r--, -wx-w-r-x. Then verify with chmod and stat.

Check: new files start at 666 minus umask, so 022 gives 644 and 077 gives 600. Directories start at 777.

## 8. Processes: ps, top, kill, killall, nice

```bash
ps aux | head
ps -ef --forest | head -20
top                                        # q to quit; try P, M, k, 1
sleep 300 &
sleep 301 &
jobs; pgrep -a sleep
kill %1                                    # or kill <pid>
killall sleep
nice -n 10 sleep 200 &
ps -o pid,ni,cmd -C sleep
renice -n 5 -p $(pgrep sleep)
sleep 400 & kill -STOP $!; ps -o pid,stat,cmd -p $!; kill -CONT $!; kill $!
```

Check: ni column shows 10 for the nice'd process. Notice kill sends SIGTERM (15) by default, and `kill -9` should be the last resort.

## 9. Disk and system info: df, du, free, uname, uptime

```bash
df
df -h
df -hT /
du -sh ~/lab
du -sh ~/lab/* | sort -h
du -h --max-depth=1 /var 2>/dev/null | sort -h | tail -5
free -h
uname -a; uname -r
uptime; uptime -p
lsblk
```

Check: sort the biggest directories under /var and compare `df -h /` used against `du -sh /` (they differ; think about why, e.g. deleted-but-open files, other mounts).

## 10. Networking: ip, ping, host, traceroute, netstat, ssh, scp

```bash
ip a                                       # ifconfig is deprecated
ip -br a
ip r
ping -c 4 google.com
host google.com
host 8.8.8.8
sudo apt install -y traceroute net-tools openssh-server
traceroute google.com
traceroute -n -m 10 google.com
netstat -tuln                              # legacy
ss -tuln                                   # modern replacement
ss -tulnp | grep ssh
sudo systemctl status ssh
ssh $USER@localhost
exit
echo "from scp" > send.txt
scp send.txt $USER@localhost:/tmp/
ls -l /tmp/send.txt
scp $USER@localhost:/tmp/send.txt ./received.txt
```

Check: if ssh refuses, `ss -tuln` should show 22 listening once openssh-server is up. For a real remote host, swap localhost for its IP. As an extension, set up key auth with `ssh-keygen -t ed25519` and `ssh-copy-id`.

## 11. Packages: apt

```bash
sudo apt update
apt list --upgradable | head
sudo apt upgrade                           # review before confirming
sudo apt install -y tree
tree -L 2 ~/lab
apt show tree | head
dpkg -L tree | head
sudo apt remove tree
sudo apt purge tree                        # also removes config files
sudo apt autoremove
```

## 12. Users: whoami, su, adduser, passwd, usermod

```bash
whoami; id
sudo adduser testuser
sudo passwd testuser
su testuser                                # check prompt and pwd
whoami; exit
sudo usermod -aG sudo testuser
id testuser
sudo usermod -aG sudo,adm testuser         # -a matters, see next line
# usermod -G sudo testuser (no -a) would REPLACE all supplementary groups
sudo -u testuser touch /tmp/by_testuser.txt; ls -l /tmp/by_testuser.txt
sudo deluser --remove-home testuser
```

Check: `su testuser` is not a login shell, `su - testuser` is. Compare `pwd` and `echo $PATH` after each.

## 13. Archives: tar, gzip, gunzip

```bash
mkdir -p arch/sub && seq 1 1000 > arch/nums.txt && cp marks.txt arch/sub/
tar -cvf backup.tar arch/
tar -tvf backup.tar                        # list without extracting
gzip backup.tar                            # produces backup.tar.gz
gunzip backup.tar.gz
tar -czvf backup.tgz arch/                 # create + gzip in one step
mkdir restore && tar -xzvf backup.tgz -C restore
tar -xvf backup.tar -C restore --strip-components=1
gzip -k nums.txt; ls -l nums.txt*
gunzip nums.txt.gz
zcat nums.txt.gz 2>/dev/null | head -2
```

Check: compare `ls -l` sizes of backup.tar and backup.tgz.

## 14. Power control: shutdown, reboot, halt (dry-run style)

Do not run these bare on a machine you are using.

```bash
sudo shutdown +10 "test"                   # schedule
shutdown --show 2>/dev/null
sudo shutdown -c                           # cancel
sudo shutdown -k now "wall message only"   # warns users, does not shut down
```

`reboot`, `halt` and `shutdown now` do what they say. Try them in a VM or throwaway container only. Know the difference: `halt` stops the OS, `poweroff` also cuts power, `shutdown` is the safe scheduler.

## 15. Echo and redirection recap: echo, >, >>

```bash
echo "Geeks for Geeks"
name=amit; echo "hi $name"; echo 'hi $name'
echo -e "a\nb\nc" > lines.txt
echo "d" >> lines.txt
echo "x" > lines.txt; cat lines.txt             # > truncates
ls /nonexistent 2> err.txt; cat err.txt
ls /nonexistent /etc > out.txt 2>&1
ls /etc | tee etc_list.txt | wc -l
```

## 16. Wrap-up challenge, no hints

Run these with a clean ~/lab2 and do not look above.

1. Create ~/lab2/logs, ~/lab2/data, ~/lab2/bin in one command.
2. Generate 500 lines in data/app.log containing a mix of INFO, WARN and ERROR using a for loop or `shuf`.
3. Show only the last 5 ERROR lines with line numbers.
4. Count lines per level using grep, then again using `sort | uniq -c`.
5. Write bin/report.sh that does step 4 and prints the result, make it executable with the smallest chmod that lets only you run and edit it, and run it.
6. Archive ~/lab2 into a gzipped tar excluding *.log, then list the contents without extracting.
7. Find every file under ~/lab2 modified in the last hour, larger than 1 KB, and print size and path.
8. Start three `sleep` processes at different nice values, show their NI column, kill them with one command.
9. Show which process listens on port 22 and its owner.
10. Report free disk space of / in human-readable form and the ten biggest items in your home directory.

## Notes on the sheet

The SkillNest page lists `ifconfig / ip a` and `netstat` as if both are current. On a modern Ubuntu, `ifconfig` and `netstat` need the net-tools package and `ss` and `ip` are the replacements. The sheet's `sed -n '1,4p'` is print-only, add `-i` to actually modify the file. `find . -name` in the sheet starts from the current directory only, use `/` for the whole system and expect permission errors without sudo.

## Cleanup

```bash
cd ~ && rm -rf ~/lab ~/lab2
```
