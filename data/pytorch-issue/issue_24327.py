Python
for (x, l, c) in zip(sources, self.loc, self.conf):
        conf.append(c(x).permute(0, 2, 3, 1).contiguous().view(c(x).size(0), -1, self.num_classes))
        loc.append(l(x).permute(0, 2, 3, 1).contiguous().view(l(x).size(0), -1, 4))