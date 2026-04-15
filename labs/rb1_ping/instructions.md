This will be some instruction for rb1_ping

Assigned commands
Host 1:
ip a 
ip addr add 10.0.0.1/24 dev eth1
ip a show eth1

Host 2:
ip a
ip addr add 10.0.0.2/24 dev eth1
ip a show eth1

Then test connectivity:
From Host 1: ping 10.0.0.2
From Host 2: ping 10.0.0.1
