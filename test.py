import autodiff as ad


a = ad.Numtor(1, name='a')
b = ad.Numtor(2, name='b')
z = a * b # mul(a,b) <- z
x = a + b # add(a,b) <- x
y = (x + z) * (x + z) # mul(add(x, z), add(x,z)) <- y
                      # add(x, z) <- <anonym>
                      # add(x, z) <- <anonym>

#z1 = a * b
#z2 = a * b
#x1 = a + b
#x2 = a + b
#y = (x1 + z1) * (x2 + z2)

# 2(a+b + a*b)*(1+b) = 2(3+2)*3 = 30
y.backward()
print(a.grad)



# a.node = register(ConstNode())
# b.node = register(VarNode())

# mul(a, b):
#     node = register(MulNode, parents=[a.node, b.node])
#     return Numtor(value=a.val*b.val, node=node)
