import os


text_path = [os.path.join(root,name) for root, dirs, files in os.walk(os.getcwd()) for name in files if "log.pooling.clustering.balanced" in name and name.endswith('.txt')]
print "==============================START======================"
for name in text_path:
    ari = []
    hs = []
    with open(name, 'r') as fp:
        for line in fp:
            line=line.strip()
            if line.startswith("Adjusted Rand Index"):
                ari.append(float(line.split()[-1]))
            elif line.startswith("Homogeneity Score"):
                hs.append(float(line.split()[-1]))
    print name
    print "ARI = %.4f" %(1.0*sum(ari)/len(ari))
    print "MS = %.4f" %(1.0*sum(hs)/len(hs))
    print "====================================================="
'''
ari = []
hs = []
with open("log.clustering.balanced.txt", 'r') as fp:
    for line in fp:
        line=line.strip()
        if line.startswith("Adjusted Rand Index"):
            ari.append(float(line.split()[-1]))
        elif line.startswith("Homogeneity Score"):
            hs.append(float(line.split()[-1]))

print 1.0*sum(ari)/len(ari)
print 1.0*sum(hs)/len(hs)
'''
