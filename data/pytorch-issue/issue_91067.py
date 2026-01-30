import boto3
ec2=boto3.resource("ec2")
rc=ec2.create_instances(ImageId="ami-031843d9eaa76ad7a",InstanceType="c5a.4xlarge",SecurityGroups=['ssh-allworld'],KeyName="nshulga-key",MinCount=1,MaxCount=1,BlockDeviceMappings=[{'DeviceName': '/dev/sda1','Ebs': {'DeleteOnTermination': True, 'VolumeSize': 150,'VolumeType': 'standard'}}])