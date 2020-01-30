# scp -i ~/.ssh/burner-west2.pem -r data something@something.compute.amazonaws.com:.
mkdir tmp
export TMPDIR=/home/ubuntu/tmp
sudo apt-get update
sudo apt-get upgrade
sudo apt-get install gcc python3-dev python3-pip libxml2-dev libxslt1-dev zlib1g-dev g++
curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py
sudo python3 get-pip.py
pip3 install virtualenv
git clone git@github.com:sdsunjay/nlp_product_reviews.git
virtualenv -p python3 nlp_product_reviews
source nlp_product_reviews/bin/activate
pip3 install -r requirements.txt
