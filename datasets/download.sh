wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies \
--no-check-certificate 'https://docs.google.com/uc?export=download&id=1ylw6AZ8AsyRmvook3FXrR2miSNajo6HE' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1ylw6AZ8AsyRmvook3FXrR2miSNajo6HE" -O patchPose.tar.gz && rm -rf /tmp/cookies.txt

tar -zxvf patchPose.tar.gz
rm -rf patchPose.tar.gz



wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies \
--no-check-certificate 'https://docs.google.com/uc?export=download&id=1KUv6PAAIuRlzoZbw1xEv4-l_3gQ3_UCF' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1KUv6PAAIuRlzoZbw1xEv4-l_3gQ3_UCF" -O hpatchesPose.tar.gz && rm -rf /tmp/cookies.txt

tar -zxvf hpatchesPose.tar.gz
rm -rf hpatchesPose.tar.gz


