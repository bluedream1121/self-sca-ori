wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies \
--no-check-certificate 'https://docs.google.com/uc?export=download&id=1qO22Gz_ktji1NOigdrHgTyZAMs4Ecbxw' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1qO22Gz_ktji1NOigdrHgTyZAMs4Ecbxw" -O weights_sca.tar.gz && rm -rf /tmp/cookies.txt

tar -zxvf weights_sca.tar.gz
rm -rf weights_sca.tar.gz


