import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Cursor, Slider
from numpy import sin
from numpy import cos
from numpy import tan
from numpy import pi
π = pi

# GEOMETRIE DE L'AILE
# nom 		valeur					description					unité

Λd			= 45 					# flèche 					(°)
c 	 		= 0.2					# corde de l'aile 			(m)
L			= 0.5					# demi-envergure de l'aile 	(m)
λ			= 5						# allongement de l'aile 	(sans unité)

# CONFIGURATION DE L'APPAREIL
αdMax		= 10					# angle d'attaque de l'aile (°)

# PARAMETRES DU CALCUL
n 			= 100					# nombre de points

# PARAMETRES GLOBAUX
S 			= 4*(L**2)/λ			# surface de l'aile 		(m²)
Λ 			= Λd*π/180				# flèche de l'aile en rad	(rad)


# AFFICHAGE DE L'AILE
xA = [0,L*tan(Λ),L*tan(Λ)+c,c,0]
yA = [0,L,L,0,0]

fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2,2)
fig.subplots_adjust(hspace=0.5)
ax1.plot(xA,yA,label='Aile')

ax1.set_xlabel('x (m)')
ax1.set_ylabel('y (m)')

# CALCUL DES COORDONNEES
# nom 		valeurs																		description

X 	= np.linspace(0.75*c + L/(2*n)*tan(Λ),	0.75*c + L*((2*n-1)/(2*n))*tan(Λ),	n)		# X et Y sont les points de
Y 	= np.linspace(L/(2*n),					L*((2*n-1)/(2*n)),					n)		# calcul des circulations

Xa 	= np.linspace(0.25*c,					0.25*c + L*(1-1/n)*tan(Λ),			n)		# A et B sont les points du
Ya 	= np.linspace(0,						L*(1-1/n),							n)		# fer à cheval
Xb 	= np.linspace(0.25*c + L/n*tan(Λ),		0.25*c + L*tan(Λ),					n)
Yb 	= np.linspace(L/n,						L,									n)

# AFFICHAGE DES POINTS
limiter		= n//40
pts			= ax1.scatter(X[::limiter],Y[::limiter]  ,label='Points de calcul des circulations',marker='+')
ptsTourbA	= ax1.scatter(Xa[::limiter],Ya[::limiter],marker='x')
ptsTourbB	= ax1.scatter(Xb[::limiter],Yb[::limiter],marker='+')
ax1.annotate("A"+str(n//2), (Xa[n//2]-0.02,Ya[n//2]+0.015))
ax1.annotate("B"+str(n//2), (Xb[n//2]-0.02,Yb[n//2]+0.015))
ax1.annotate("P"+str(n//2), (X[n//2]-0.02,Y[n//2]+0.015))

ax1.legend()
ax1.set_title('Répartition des points pour n=' + str(n) + ' segments élémentaires')

# CALCUL DES VITESSES
def wi(x,y,xa,ya,xb,yb):
	# Valable seulement si (x,y) est dans le plan du tourbillon --> diedre = 0
	resAB	= (((xb-xa)*(x-xa)+(yb-ya)*(y-ya))/(((x-xa)**2 + (y-ya)**2)**0.5))
	resAB 	= resAB - (((xb-xa)*(x-xb)+(yb-ya)*(y-yb))/(((x-xb)**2 + (y-yb)**2)**0.5))
	resAB	= resAB/((x-xa)*(y-yb)-(x-xb)*(y-ya))

	resA	= 1 + (x-xa)/(((x-xa)**2 + (y-ya)**2)**0.5)
	resA 	= -resA/(y-ya)

	resB 	= 1 + (x-xb)/(((x-xb)**2 + (y-yb)**2)**0.5)
	resB 	= resB/(y-yb)

	result 	= resAB + resA + resB
	return result

C  = np.zeros((n,n))
Cp = np.zeros((n,n))
for i in range(0,n):
	for j in range(0,n):
		C[i][j] = wi(X[i],Y[i],Xa[j],Ya[j],Xb[j],Yb[j])
		Cp[i][j] = wi(X[i],Y[i],Xb[j],-Yb[j],Xa[j],-Ya[j]) #ne pas oublier d'inverser A et B pour le sens du tourbillon

# RESOLUTION DU PROBLEME
# On a (C+Cp)Γ = -4παU
# On résoud Γ = γ*(-4παU)
I = np.ones(n)
M = np.linalg.inv(C+Cp)

γ = -M.dot(I)

A = np.linspace(0,αdMax,2)
Cz = [4/S*L/n*4*π*α*π/180*np.sum(γ) for α in A]

line2, = ax2.plot(A,Cz)
ax2.set_xlabel('α (°)')
ax2.set_ylabel('Cz')
ax2.set_title("Cz fonction de l'angle d'attaque de l'aile")

class DataCursor(object):
    text_template = 'x: %0.2f\ny: %0.2f'
    x, y = 0.0, 0.0
    xoffset, yoffset = -20, 20
    text_template = 'x: %0.2f\ny: %0.2f'

    def __init__(self, ax):
        self.ax = ax
        self.annotation = ax.annotate(self.text_template, 
                xy=(self.x, self.y), xytext=(self.xoffset, self.yoffset), 
                textcoords='offset points', ha='right', va='bottom',
                bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.5),
                arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0')
                )
        self.annotation.set_visible(False)

    def __call__(self, event):
        self.event = event
        # xdata, ydata = event.artist.get_data()
        # self.x, self.y = xdata[event.ind], ydata[event.ind]
        self.x, self.y = event.mouseevent.xdata, event.mouseevent.ydata
        if self.x is not None:
            self.annotation.xy = self.x, self.y
            self.annotation.set_text(self.text_template % (self.x, self.y))
            self.annotation.set_visible(True)
            event.canvas.draw()

fig.canvas.mpl_connect('pick_event', DataCursor(plt.gca()))
line2.set_picker(5)



#Tracé des Gamma en fonction de y

def get_γ(𝛿):
	X 	= np.linspace(0.75*c + L/(2*n)*tan(𝛿*π/180),		0.75*c + L*((2*n-1)/(2*n))*tan(𝛿*π/180),		n)		# X et Y sont les points de
	Y 	= np.linspace(L/(2*n),							L*((2*n-1)/(2*n)),							n)		# calcul des circulations

	Xa 	= np.linspace(0.25*c,							0.25*c + L*(1-1/n)*tan(𝛿*π/180),				n)		# A et B sont les points du
	Ya 	= np.linspace(0,								L*(1-1/n),									n)		# fer à cheval
	Xb 	= np.linspace(0.25*c + L/n*tan(𝛿*π/180),			0.25*c + L*tan(𝛿*π/180),						n)
	Yb 	= np.linspace(L/n,								L,											n)
	C  = np.zeros((n,n))
	Cp = np.zeros((n,n))
	for i in range(0,n):
		for j in range(0,n):
			C[i][j] = wi(X[i],Y[i],Xa[j],Ya[j],Xb[j],Yb[j])
			Cp[i][j] = wi(X[i],Y[i],Xb[j],-Yb[j],Xa[j],-Ya[j]) #ne pas oublier d'inverser A et B pour le sens du tourbillon
	I = np.ones(n)
	M = np.linalg.inv(C+Cp)

	γ = -M.dot(I)
	return γ
for 𝛿 in [-60,-45, 0, 45, 60]:
	ax3.plot(Y,get_γ(𝛿),label="flèche="+str(𝛿))

ax3.set_xlabel('y (m)')
ax3.set_ylabel('γ (m^2/s)')
ax3.set_title("Répartition de la Circulation en fonction de la position horizontale sur l'aile")
ax3.legend()


# Variation de Γ en fonction de la flèche à L, et c constants
Λ = np.linspace(-60,60,25)
Γ = []
for 𝛿 in Λ:
	Γ.append(L/n*np.sum(get_γ(𝛿)))
ax4.plot(Λ,Γ)
ax4.set_xlabel("Flèche de l'aile (°)")
ax4.set_ylabel('∫Γdy (ΣγiΔy)')
ax4.set_title("Circulation en fonction de la flèche de l'aile")
plt.show()