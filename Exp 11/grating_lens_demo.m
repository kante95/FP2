%creating a sawtooth grating and a diffractive lens with MATLAB


%creating coordinate system
N=1024; %size of grid
pixsize=8e-6; %pixelsize of SLM
x=(1:N)*pixsize;
y=x;

%% 1) sawtooth grating
[X,Y]=ndgrid(x,y);

%% 1) sawtooth grating

Gx=50*pixsize; %period of x-grating
kx=2*pi/Gx; %kx-vector
Gy=25*pixsize; %period of y-grating
ky=2*pi/Gx; %ky-vector

grating=angle(exp(1i*(kx*X+ky*Y)));
imagesc(grating); axis equal; axis tight;

%alternatively use the "mod" function: 
grating=mod(kx*X+ky*Y,2*pi);

%% 2) diffractive lens
figure()

f=300e-3; %focal length in meter
lambda=633e-9; %wavelength in meter
xc=N/2*pixsize; %center coordinates
yc=xc;
lens=angle(exp(-1i*pi/lambda/f*((X-xc).^2+(Y-yc).^2)));
imagesc(lens); axis equal; axis tight;