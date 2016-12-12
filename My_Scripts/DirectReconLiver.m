% Direct Reconstruction of Pharmakokinetics in Liver using the Patlack
% model. 

clear all;
clc;
clf;

%% Loading Data
% load data from GRASP, NYU
load('C:\Users\Kall\Desktop\PhD\DATA\Matlab Scripts\XDGRASP_Demo\Dataset\data_DCE.mat') 
% kc is respiratory motion signal
% kdata  (readouts - spokes - coils ) is the k-space data. It is only one slice selected from the 3D
% stack-of-stars
% b1 is the coil sensitivity maps of the selected slice 
% k is the radial k-space trajectory and w is the corresponding density compensation

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Coil sensitivity maps are computed with the adaptive array-combination
%technique (33,34) using coil-reference data from the temporal
%average of all acquired spokes. 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

[nx,ntviews,nc]=size(kdata);
%nx: readout point of each spoke (2x oversampling included)
%ntviews: number of acquired spokes
%nc: number of coil elements

%% Data Sorting
nline = 34; % number of spokes in each reconstructed image (contrast-enhanced phase) - Option 1
nt = floor(ntviews/nline); %number of images (constrast-enhanced phases) 
 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% d = argmin{ || F S d - m ||22 + ë|| T d ||1} 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%A ramp filter in the kx-ky plane was applied to each spoke to compensate for variable density sampling.
% plot(sqrt(w))  %to see the filter. 
kdata=kdata.*repmat(sqrt(w),[1,1,nc]);

clear kdata_u k_u w_u Res_Signal_u 
for ii=1:nt                          % sorting the spokes into frames kdata_u (readout, spokes, coils, frame)
    kdata_u(:,:,:,ii)=kdata(:,(ii-1)*nline+1:ii*nline,:); 
    k_u(:,:,ii)=k(:,(ii-1)*nline+1:ii*nline);  % k_u ( readout trajectories, spokes, frame)
    w_u(:,:,ii)=w(:,(ii-1)*nline+1:ii*nline);
end

%% Reconstruction

% Multicoil NUFFT operator
% Based on the NUFFT toolbox from Jeff Fessler
% Input
% k: k-space trajectory
% w: density compensation
% b1: coil sensitivity maps
param.E=MCNUFFT(double(k_u),double(w_u),double(b1)); 
param.y=double(kdata_u);
recon_cs=param.E'*param.y;
data_gridding=recon_cs/max(abs(recon_cs(:)));

fftliver = fft2(data_gridding); 


ns=1; % choose one slice of k-space
kliver(:,:,ns,:)=fftliver(:,:,:);
kliver = repmat(kliver, [1,1,1,1,8]);
opt.size=size(kliver);  
[kx,ky,kz,nt,ncoil]=size(kliver);

%sMaps=sMaps(:,:,ns,:,:); % Sensitivity Maps ???  
%sMaps=reshape(sMaps,[kx ky 1 1 ncoil]); % returns an n-dimensional array with the same elements as A but reshaped to have the size m-by-n-by-p-by-.. it doesn't change something in this case ???
sMaps=ones(kx,ky,ns,1,8);
sMaps=reshape(sMaps,[kx ky 1 1 ncoil]);

if ~exist('R1','var')  % use simulated uniform M0 and R1 if none exists
M0=5*ones(kx,ky,'single'); %use simulated M0, R1 // M0 is the equilibrium longitudial magnetization
R1=1*ones(kx,ky,'single');  % R1 is the precontast R1 (reciprocal of T1)
end

 imgF = sum(ifft2(kliver(:,:,1,1:4,1)),4);
 imgF = repmat(imgF,[1,1,1,nt]);

 %% set parameters
opt.wname='db4'; % wavelet parameters
opt.worder={[1 2],[1 2],[1 2]};

opt.R1=R1;
opt.M0=M0;
opt.Sb=repmat(imgF(:,:,:,1),[1 1 1 nt]);  %baseline image // the 1st dynamic propagate to all dynamics
opt.alpha=pi*15/180; %flip angle // = 15o 
opt.TR=0.006;  %TR // = 6 ms 

delay=8; % delay frames for contrast injection
tpres=5/60; % temporal resolution, unit in seconds! // ???
opt.time=[zeros(1,delay),[1:(nt-delay)]*tpres]; 
opt.plot=1;  % to plot intermediate images during recon


opt.lambdaA=[0.000 0.000 0.000 0.000]; % Kt:TV, Wavelet, Vp: TV, wavelet
opt.Initer=10;  %inter interations
opt.Outiter=10; % outer interations

%% calculate fully-sampled Ktrans and Vp
CONCF = sig2conc2(real(imgF),R1,M0,opt.alpha,opt.TR);  % functon to calculate contrast concentration from image intensity
opt.AIF=SAIF_p(opt.time); % get population-averaged AIF
[Kt,Vp]=conc2Ktrans(CONCF,opt.time,opt.AIF);

% U1=reshape(nshift(U11>0,[1 2]),[kx,ky,1,nt]);
% U1=repmat(U1,[1 1 1 1 ncoil]);
U1=ones(1,1,1,1,1);  

imgU= ifft2(kliver(:,:,:,:,1));
CONC1 = sig2conc2(real(imgU),R1,M0,opt.alpha,opt.TR);
[Kt_U,Vp_U]=conc2Ktrans(CONC1,opt.time,opt.AIF);

%% Direct reconstruction
% use zeros as initial guess
Kt_1=zeros(opt.size(1),opt.size(2)); 
Vp_1=Kt_1;

tic,
[Kt_r1,Vp_r1,iter,fres]=Kt_Vp_SEN(Kt_1,Vp_1,sMaps,U1,kliver,opt); % This is the function to alternatively reconstruct Ktrans and Vp using a small number of l-BFGS iterations
toc,

iter,
Kt_r1=real(Kt_r1);  % get real value after recon
Vp_r1=real(Vp_r1);

%% display results
figure;
subplot(2,3,1); imagesc(Kt,[0 0.1]); title('Fully-sampled Ktrans'); colorbar;axis image; axis off;
subplot(2,3,2); imagesc(Kt_U,[0 0.1]); title('Zero-padded Ktrans'); colorbar;axis image; axis off;
subplot(2,3,3); imagesc(Kt_r1,[0 0.1]); title('Reconstructed Ktrans'); colorbar;axis image; axis off;
subplot(2,3,4); imagesc(Vp,[0 0.2]); title('Fully-sampled Vp'); colorbar;axis image; axis off;
subplot(2,3,5); imagesc(Vp_U,[0 0.2]); title('Zero-padded Vp'); colorbar;axis image; axis off;
subplot(2,3,6); imagesc(Vp_r1,[0 0.2]); title('Reconstructed Vp');  colorbar;colorbar;axis image; axis off;

figure;
plot(fres);title('Objective function changes over iteration');
xlabel('Iteration'); ylabel('Objective function value');
