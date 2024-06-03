# Steps to download NOAA NWS OWP FIM products
Last updated 2/20/2024 - Scott Lawson

1. Open the start menu and open a new Command Prompt.
2. Paste the following command and hit enter to install the AWS CLI (command line interface)
```
msiexec.exe /i https://awscli.amazonaws.com/AWSCLIV2.msi
```
Follow the instructions in the install wizard.  Once installation is complete, close the Command Prompt.

2.  Create a new folder where you want to download your hydrofabric.  For the example, I'll use:  C:\Users\Kensf\Downloads\aws_fim_test
3. Open a new Command Prompt, and change directory to your new folder
```
cd C:\Users\Kensf\Downloads\aws_fim_test
```
4. Type in 
```
aws s3 sync s3://noaa-nws-owp-fim/hand_fim/fim_4_4_0_0/02020001/ . --request-payer requester
```
Where 02020001 is the HUC8 you are interested in.

(troubleshooting) I'm guessing newer version of FIM will be released after I write this.  To find the correct bucket, execute
```
aws s3 ls s3://noaa-nws-owp-fim/hand_fim/ --request-payer requester 
```
and replace "fim_4_4_0_0" with the newer bucket.