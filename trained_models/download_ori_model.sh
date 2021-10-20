wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies \
--no-check-certificate 'https://docs.google.com/uc?export=download&id=1bCZqtRg2LoMQuZApagVLf3sd5daeW70A' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1bCZqtRg2LoMQuZApagVLf3sd5daeW70A" -O weights_ori.tar.gz && rm -rf /tmp/cookies.txt

tar -zxvf weights_ori.tar.gz
rm -rf weights_ori.tar.gz
