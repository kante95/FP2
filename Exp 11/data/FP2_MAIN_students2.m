%% Settings
%to run single sections of code (marked by %%) use strg+enter
clear all; 
close all;

imageDim = [1000 1000]; %dimensions of the image, for part 5 needs to be nxn
windowPosition = [1920+1920/2-imageDim(1)/2 0]; %position of the window on the SLM (works as a second screen)
diskRadius = 500;
Phase_off=0;

%% Part 1: grating

periodX = 40; %grating period in pixels in x-direction
periodY = inf; %grating period in pixels in y-direction

[X,Y]=meshgrid(1:imageDim(1),1:imageDim(2)); %RAUSLÖSCHEN
grating = mod(X,periodX);


figure(1)
imagesc(grating)
imageProperties;

%g1 = fun_grating(grating_periodX, grating_periodY, imageDim); %g(rating) is a phase function between 0 and 2*pi 


%% Part 2: Lens

f = 0.2; %focal length of the lens in meters
z0 = -0.15; %shift of the focal point due to the SLM in meters
pixsize = 8e-6;
[X,Y]=meshgrid((-imageDim(1)/2+1:imageDim(1)/2)*pixsize,(-imageDim(2)/2+1:imageDim(2)/2)*pixsize); %RAUSLÖSCHEN
lambda = 0.633e-6;

%lens = 
phase = mod((pi/(lambda*f^2))*(X.^2+Y.^2)*z0,2*pi); 

figure(1)
imagesc(phase,[0,2*pi]);
imageProperties;
%l = (fun_lens(imageDim,focalPoint,z0));  %l(ens) is a phase function between 0 and 2*pi

%% Part 3: 
% a x-grating shift a point alog the x-axis, how can one 
% produce another point in a different axis --> multiply or add phases?

% implement zwo focal points using the function
% grating(periodX, periodY, ImageDimension).. 
g2= angle(exp(1i*fun_lens(imageDim,f,0.01)+1i*fun_grating(-50,-20,imageDim))+exp(1i*fun_grating(-30,20,imageDim)));

figure(1)
imagesc(g2,[-pi,pi])
imageProperties;

%% Part 4: generate a hologram

% load image (gif, jpg or png)
%[a,b]=uigetfile({'*.gif;*.jpg;*.png'},'load image');
%disp(['select: ', fullfile(a, b)])
%datei = fullfile(b, a);


datei='Gruppe 2\francisdrake.jpg';

% resize image to fit imageDim
bild=imresize(im2double(sum(imread(datei),3)),imageDim);

% Gerchberg Saxton algorithm, shows iteration steps
iterSteps = 10; %numer of iterations
gsData=gerchbergSaxton(bild, iterSteps, imageDim,true);

% show phase image on SLM
figure(1);
imagesc( imrotate((angle(gsData.*exp(j*fun_grating(inf,inf,imageDim)))),180),[-pi,pi] )
imageProperties;


%% Part 5: adaptive correction with SLM

% take three images (camera) with different phase shifts: 0, 2pi/3, 4pi/3
% shifting the grating by one pixel results in a phase shift of 2pi/G, where
% G is the grating period

%% Part 5.1: take images with phi = 0, 2pi/3, 4pi/3
%save images as png!

grating_PeriodX = 3; %grating period in pixels in x-direction
diskRadius=250; %size of area that shows the grating (circle)
g = fun_grating(grating_PeriodX,inf, imageDim);
g_masked = maskCircle(g,diskRadius, [0 0]);

gratingShift = 0;% -->shift grating

g_masked_shifted=circshift(g_masked,[0 gratingShift]); 
figure(1);
imagesc(g_masked_shifted,[0 2*pi]);
imageProperties;

%% Part 5.2: import images into matlab

clear bild;
disp('Select the three images with different phases (in increasing order!)')
[a,b]=uigetfile({'*.gif;*.jpg;*.png'},'load image','MultiSelect', 'on');
for m=1:3
   bild(:,:,m)=(im2double(sum(imread(fullfile(b, char(a(m)))),3)));
end
a %check if images are in increasing order

%% interactive resizing of the image
%  double-klick to exit interactive modus

figure(2);
clear subimage;
[subimage(:,:,1) rect_coord] = imcrop(bild(:,:,1)/255);
%webcamDiskRadius=mean(size(subimage))/2;
imagesc(subimage);
subimage(:,:,2)=imcrop(bild(:,:,2)/255,rect_coord);
subimage(:,:,3)=imcrop(bild(:,:,3)/255,rect_coord);

%% Part 5.3: corrections

phases = [0 2*pi/3 4*pi/3];
ph = zeros ( size(subimage(:,:,1) ));
for m=1:3
    ph = ph +  subimage(:,:,m) * exp(phases(m)*1i);
end

figure(13)
imagesc(angle(ph), [-pi pi]); colormap gray;

 %last number in imrotate corrects a possible rotation of the SLM or camera 
 % --> change if necessary
phcutted = imrotate(angle(ph),0,'crop');

%phcutted = -(mod(angle(ph), 2*pi));
figure(2);
imagesc(-phcutted,[-pi pi]); 
colormap gray

%% rescale images to the size of the hologram

figure(2)
webcamDiskRadius=mean(size(phcutted))/2;
phscaled=imresize(phcutted, diskRadius/webcamDiskRadius, 'cubic');
imagesc(phscaled)
axis image
axis off
colormap('gray')

size(phscaled)


%% superimposing the image with a grating

grating_PeriodX = 3; %grating period of superimposed grating (does not have to be the same as in 5.1)

g = fun_grating(grating_PeriodX,inf,imageDim);
gs = size(g);
gsm = gs/2;

psm = size(phscaled)/2;

zeroed = zeros(imageDim);
zeroed(floor(gsm(1)-psm(1)) : floor(gsm(1)+psm(1)-1), floor(gsm(2)-psm(2)) : floor(gsm(2)+psm(2)-1))=-phscaled;
max(max(zeroed))
combined = mod((g+zeroed),2*pi);

figure(2);
imagesc(combined,[0 2*pi])
axis image
axis off
colormap('gray')

figure(1)
imagesc(combined,[0 2*pi])
axis image
axis off
colormap('gray')
imageProperties;