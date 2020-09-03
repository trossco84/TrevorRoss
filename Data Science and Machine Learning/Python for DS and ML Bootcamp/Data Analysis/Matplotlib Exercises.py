import numpy as np 
import matplotlib.pyplot as plt 

x = np.arange(0,100)
y=x*2
z=x**2

plt.show

#Exercise1
fig1 = plt.figure()
ax1 = fig1.add_axes([0,0,1,1])
ax1.set_title('title')
ax1.set_xlabel('x')
ax1.set_ylabel('y')
ax1.plot(x,y)
fig1

#Exercise2
fig2 = plt.figure()
ax2_1 = fig2.add_axes([0,0,1,1])
ax2_2 = fig2.add_axes([0.2,0.5,0.2,0.2])
ax2_1.set_xlabel('x')
ax2_1.set_ylabel('y')
ax2_2.set_xlabel('x')
ax2_2.set_ylabel('y')
ax2_1.plot(x,y)
ax2_2.plot(x,y)
fig2

#Exercise3
fig3 = plt.figure()
ax3_1 = fig3.add_axes([0,0,1,1])
ax3_2 = fig3.add_axes([0.2,0.5,0.4,0.4])

ax3_1.set_xlabel('X')
ax3_1.set_ylabel('Z')
ax3_1.plot(x,z)

ax3_2.set_ylabel('Y')
ax3_2.set_xlabel('X')
ax3_2.set_title('zoom')
ax3_2.set_xlim([20,22])
ax3_2.set_ylim([30,50])
ax3_2.plot(x,y)

#Exercise4
fig4,ax4 = plt.subplots(nrows=1,ncols=2,figsize=(8,2))
ax4[0].plot(x,y,color='blue',ls='--')
ax4[0].set_xlabel('x')
ax4[0].set_ylabel('y')
ax4[1].plot(x,z,color='red',lw=5)
ax4[1].set_xlabel('x')
ax4[1].set_ylabel('z')
ax4[1].yaxis.labelpad=-8
