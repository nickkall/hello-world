 clear all

load kspace_liver.mat
%load baseline_image.mat

ns=1;
k = fft2(data_grasp); % convert image space into k-space 
imshow(abs(k(:,:,1))); % k(x,y,dynamic images (34 spokes grouped)) 
kliver(:,:,1,:)=k(:,:,:);  % transform to be in the same format as in the script from Kt_Vp_SEN_AD_3d.m
kliver = repmat(kliver, [1,1,1,1,8]); % transform to be in the same format as in the script from Kt_Vp_SEN_AD_3d.m
k = kliver;   % k (x,y,z,dynamics,coils)

opt.size=size(k);  
[kx,ky,kz,nt,ncoil]=size(k);

sMaps=ones(kx,ky,ns,1,8); %sensitivity maps to be consisten with the format of the script. Assume that is ones as the sensitivity is already in the reconstructed images from GRASP
sMaps=sMaps./size(sMaps,5); %Normalization for the sum function later.
sMaps=reshape(sMaps,[kx ky 1 1 ncoil]);

if ~exist('R1','var')  % use simulated uniform M0 and R1 if none exists
M0=5*ones(kx,ky,'single'); %use simulated M0, R1 // M0 is the equilibrium longitudial magnetization
R1=1*ones(kx,ky,'single');  % R1 is the precontast R1 (reciprocal of T1)
end

%imgF2 = ifft2(k(:,:,1,:,1));
imgF=sum(conj(repmat(sMaps,[1 1 1 nt 1])).*ifft2(k),5);  % get "fully-sampled" // image ( 256   256     1    32)
% baseline_image = mean(data_grasp_baseline,3);
% baseline_image = repmat(baseline_image,[1,1,1,nt]);

%% set parameters
opt.wname='db4'; % wavelet parameters
opt.worder={[1 2],[1 2],[1 2]};

opt.R1=R1;
opt.M0=M0;
base_line_image = mean(imgF(:,:,1,1:4),4);
opt.Sb = repmat(base_line_image,[1 1 1 nt]);
%opt.Sb=repmat(imgF(:,:,:,1),[1 1 1 nt]);  %baseline image // the 1st dynamic propagate to all dynamics. It has to be fully-sampled... 
% baseline_image = mean(data_grasp_baseline,3);
% baseline_image = repmat(baseline_image,[1,1,1,nt]);
%opt.Sb = baseline_image;
opt.alpha=pi*12/180; %flip angle // = 12 
opt.TR=0.00383;  %TR // = 6 ms 

delay=5; % delay frames for contrast injection
tpres=4.86/60; % temporal resolution, unit in seconds! // ???
opt.time=[zeros(1,delay),[1:(nt-delay)]*tpres]; 
opt.plot=1;  % to plot intermediate images during recon

opt.lambdaA=[0.000 0.000 0.000 0.000]; % Kt:TV, Wavelet, Vp: TV, wavelet
opt.Initer=10;  %inter interations
opt.Outiter=10; % outer interations
opt.Rcs = 4.3; % Relaxivity for Magnevist @ 1.5T Ref: T1 relaxivities of gadolinium-based magnetic resonance contrast agents in human whole blood at 1.5, 3, and 7 T.
%% calculate fully-sampled Ktrans and Vp
CONCF = sig2conc2(abs(imgF),R1,M0,opt.alpha,opt.TR,abs(opt.Sb));  % functon to calculate contrast concentration from image intensity
opt.AIF=SAIF_p(opt.time); % get population-averaged AIF
[Kt,Vp]=conc2Ktrans(CONCF,opt.time,opt.AIF);

% figure;
% subplot(2,3,1); imagesc(Kt,[0 0.1]); title('Fully-sampled Ktrans'); colorbar;axis image; axis off;
%% undersamping by RGR
U1=ones(kx,ky,1,nt,ncoil);
kU = k.*U1;
k_Baseline = fft2(opt.Sb);
kU(:,:,:,1,:) = repmat(k_Baseline(:,:,:,1,:), [1 1 1 1 ncoil]);  % Fist image is fullysampled (replace with the baseline_image)
%imgU = ifft2(kU(:,:,1,:,1));
%isequal(abs(imgF-imgU)<=eps,ones(kx,ky,kz,nt)) % check equality
imgU=sum(conj(repmat(sMaps,[1 1 1 nt 1])).*ifft2(k),5);
CONC1 = sig2conc2(abs(imgU),R1,M0,opt.alpha,opt.TR);
[Kt_U,Vp_U]=conc2Ktrans(CONC1,opt.time,opt.AIF);

%% Direct reconstruction
% use zeros as initial guess
Kt_1=zeros(opt.size(1),opt.size(2)); 
Vp_1=Kt_1;

tic,
[Kt_r1,Vp_r1,iter,fres]=Kt_Vp_SEN(Kt_1,Vp_1,sMaps,U1,kU,opt); % This is the function to alternatively reconstruct Ktrans and Vp using a small number of l-BFGS iterations
toc,

iter,
Kt_r1=abs(Kt_r1);  % get real value after recon
Vp_r1=abs(Vp_r1);

%% display results
figure;
subplot(2,3,1); imagesc(abs(Kt),[0 0.1]); title('Fully-sampled Ktrans'); colorbar;axis image; axis off;
subplot(2,3,2); imagesc(abs(Kt_U),[0 0.1]); title('Zero-padded Ktrans'); colorbar;axis image; axis off;
subplot(2,3,3); imagesc(abs(Kt_r1),[0 0.1]); title('Reconstructed Ktrans'); colorbar;axis image; axis off;
subplot(2,3,4); imagesc(abs(Vp),[0 0.2]); title('Fully-sampled Vp'); colorbar;axis image; axis off;
subplot(2,3,5); imagesc(abs(Vp_U),[0 0.2]); title('Zero-padded Vp'); colorbar;axis image; axis off;
subplot(2,3,6); imagesc(abs(Vp_r1),[0 0.2]); title('Reconstructed Vp');  colorbar;colorbar;axis image; axis off;

figure;
plot(fres);title('Objective function changes over iteration');
xlabel('Iteration'); ylabel('Objective function value');
