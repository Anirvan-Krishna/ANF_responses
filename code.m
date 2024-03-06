%% Neuronal Coding of Sensory Information (EC60004)
%% Project-1 | Anirvan Krishna | Roll no. 21EE30004


%% Part-A
%% Question-1: Response to Tones (Rate Representation)

% Initializations
bf = [500, 4000]; % Best frequency 500 Hz and 4 KHz
intensities = (-10:10:80); % Sound Pressure Level (SPL) in dB

% Setting up tone frequencies
h = (0:1/8:7);
tones = zeros(1,57);

for i = 1:57
    tones(i) = 125*(2.0^h(i)); % Calculate tone frequencies
end

ramp_time = 10e-03; % Ramp time = 10ms
duration = 200e-03; % Stimulus time duration = 200ms
Fs = 100e03; % Sampling frequency of 100 KHz

%% PSTH Parameters
repititions = 10; % Number of repetitions of stimulus
%psthbinwidth = 0.5e-3; % Bin width in seconds
irpts = ramp_time*Fs;
t = 0:1/Fs:duration-1/Fs; % Time vector
mxpts = length(t);

rate_matrix = zeros(57, 10, 2); % Matrix to store rates (frequency, intensity, BF)

%% Creating the stimuli and generating output

for i = 1:2 % Loop through 2 Best Frequencies (BFs)

    % Model fiber parameters
    CF = bf(uint8(i)); % Characteristic Frequency (CF) in Hz   
    cohc = 1.0; % Normal outer hair cell (OHC) function
    cihc = 1.0; % Normal inner hair cell (IHC) function
    fiberType = 3; % Spontaneous rate (in spikes/s) of the fiber BEFORE refractory effects;
                   % "1" = Low; "2" = Medium; "3" = High
    implnt = 0; % "0" for approximate or "1" for actual implementation of the power-law functions in the Synapse
    
    for j = 1:10 % 10 repetitions for each BF
        
        % Stimulus parameters
        stimdb = intensities(j); % Stimulus intensity in dB SPL
            
        for k = 1:57
                
                disp("Rep - " + j + ". Freq - " + k);
                
                F0 = tones(k); % Stimulus frequency in Hz   
    
                pin = sqrt(2)*20e-6*10^(stimdb/20)*sin(2*pi*F0*t); % Unramped stimulus
                pin(1:irpts)=pin(1:irpts).*(0:(irpts-1))/irpts; 
                pin((mxpts-irpts):mxpts)=pin((mxpts-irpts):mxpts).*(irpts:-1:0)/irpts;
    
                vihc = catmodel_IHC(pin,CF,repititions,1/Fs,duration*2,cohc,cihc); 
                [synout,psth] = catmodel_Synapse(vihc,CF,repititions,1/Fs,fiberType,implnt); 
                
                rate_matrix(k,j,i) = sum(psth(1:(length(psth)/2))); % Calculate rate and store in matrix
                timePSTH = length(psth)/Fs;
               
        end
    end
end
rate_matrix = rate_matrix / repititions;
rate_matrix = rate_matrix / ((length(psth)/Fs)/2);

%% Plotting Rate vs Frequency graphs
intensities = (-10:10:80);
for i = 1:2
    figure;
    grid on;
    hold on;
    
    for j = 1:10
        semilogx(tones, squeeze(rate_matrix(:,j,i)));
    end
    xlabel('Frequency');
    ylabel('Rate');
    title(['Rate vs Frequency ', num2str(bf(i)), 'Hz ANF']);
    hold off;
end
%% Plotting Rate vs Intensity curves
rate_int = zeros(10,2);
for i = 1:2
    for j = 1:10
        rate_int(j,i) = rate_matrix(25*(i-1)+1,j,i);
    end
end
fig = figure;
hold on;
grid on;
plot((-10:10:80), rate_int(:,1));
plot((-10:10:80), rate_int(:,2));
xlabel('Intensity');
ylabel('Rate');
title('Rate vs Intensity for both ANFs');
legend('500 Hz', '4000 Hz');

%% Question-2:
bf = 600; % Best frequency 600 Hz
intensities = (-20:5:80); % Sound Pressure Level (SPL) in dB

% Setting up tone frequencies
h = (0:1/8:6);
tones = zeros(1,49);

for i = 1:49
    tones(i) = 125*(2.0^h(i)); % Calculate tone frequencies
end

ramp_time = 10e-03; % Ramp time = 10ms
duration = 200e-03; % Stimulus time duration = 200ms
Fs = 100e03; % Sampling frequency of 100 KHz

%% PSTH Parameters
repititions = 10; % Number of repetitions of stimulus
psthbinwidth = 0.5e-3; % Bin width in seconds
irpts = ramp_time*Fs;
t = 0:1/Fs:duration-1/Fs; % Time vector
mxpts = length(t);

rate_matrix = zeros(49, 21); % Matrix to store rates (frequency, intensity)

%% Creating the stimuli and generating output

% Model fiber parameters
CF = bf; % Characteristic Frequency (CF) in Hz   
cohc = 1.0; % Normal outer hair cell (OHC) function
cihc = 1.0; % Normal inner hair cell (IHC) function
fiberType = 3; % Spontaneous rate (in spikes/s) of the fiber BEFORE refractory effects;
                   % "1" = Low; "2" = Medium; "3" = High
implnt = 0; % "0" for approximate or "1" for actual implementation of the power-law functions in the Synapse
    
for j = 1:21 % 10 repetitions for each BF
        
        % Stimulus parameters
        stimdb = intensities(j); % Stimulus intensity in dB SPL
            
        for k = 1:49
                
                disp("Rep - " + j + ". Freq - " + k);
                
                F0 = tones(k); % Stimulus frequency in Hz   
    
                pin = sqrt(2)*20e-6*10^(stimdb/20)*sin(2*pi*F0*t); % Unramped stimulus
                pin(1:irpts)=pin(1:irpts).*(0:(irpts-1))/irpts; 
                pin((mxpts-irpts):mxpts)=pin((mxpts-irpts):mxpts).*(irpts:-1:0)/irpts;
    
                vihc = catmodel_IHC(pin,CF,repititions,1/Fs,duration*2,cohc,cihc); 
                [synout,psth] = catmodel_Synapse(vihc,CF,repititions,1/Fs,fiberType,implnt); 
                
                rate_matrix(k,j) = sum(psth(1:(length(psth)/2))); % Calculate rate and store in matrix
                timePSTH = length(psth)/Fs;
               
        end
end

%% Separating out 'ah', generating different intensity stimuli 

h = (0:1/8:6); % Adjusted to span 6 octaves
ANF = zeros(1,length(h));
for k = 1:length(h)
    ANF(k) = 125*(2^h(k)); % Adjusted starting frequency to 125 Hz
end

% intensities = -20:5:80; % Intensities specified in the first problem statement
[fivewo, fs] = audioread('fivewo.wav');
fivewo = fivewo';
S = 1+round(1.05*fs);
E = 1+round(1.15*fs);
L = length(fivewo);
[y,fs] = audioread('fivewo.wav',[1+round(1.05*fs),1+round(1.15*fs)]);
y = y';
Fs = 100e3;
rt = 10e-3;
%sound(y,Fs);
tester = [fivewo(1:S) zeros(1,50000) fivewo(S:E) zeros(1,50000) fivewo(E:L)] ;
sound(tester,Fs);
%t = 0:1/Fs:0.2*fs/Fs; % time vector
t = (0:length(y)-1)/Fs;
mxpts = length(t);
irpts = rt*Fs;
y(1:irpts) = y(1:irpts) .* ((0:(irpts-1))/irpts); 
y((mxpts-irpts):mxpts) = y((mxpts-irpts):mxpts).*((irpts:-1:0)/irpts);
%s = size(t);
% x = rms(y);
% I_rms = 20*log10(x/(20*10^(-6)));
rate_matrix2 = zeros(1,21);
% Input = zeros(1,length(y));
cohc  = 1.0;   % normal ohc function
cihc  = 1.0;   % normal ihc function
fiberType = 3;
implnt = 0;
nrep = 80; % Adjusted repetitions as per the first problem statement
CF = 600; % Adjusted to 600 Hz as per the first problem statement

% psthbinwidth = 0.5e-3;
%psthbins = round(psthbinwidth*Fs);  % number of psth bins per psth bin
ahTime = length(y) / Fs ;
for i = 1:21 
    Input = y * 10^(((i-1)*10 - 95)/20);
    vihc = catmodel_IHC(Input,CF,nrep,1/Fs,ahTime*2,cohc,cihc); 
    [synout,psth] = catmodel_Synapse(vihc,CF,nrep,1/Fs,fiberType,implnt); 
    %{
    timeout = (1:length(psth))*1/Fs;
    psthtime = timeout(1:psthbins:end); % time vector for psth
    pr = sum(reshape(psth,psthbins,length(psth)/psthbins))/nrep; % pr of spike in each bin
    Psth = pr/psthbinwidth; % psth in units of spikes/s
    %}
    rate_matrix2(i) = sum(psth(1:(length(psth)/2)));
end
rate_matrix2 = rate_matrix2 / nrep;
rate_matrix2 = rate_matrix2 / ahTime ;


%% Plotting Rate vs Intensity of 'ah' and that of 600 Hz ANF for stimulus at BF

fig = figure;
grid on;
hold on;
plot((-20:5:80), rate_matrix2);         % for the ah stimulus
plot((-20:5:80), rate_matrix(19,:));   % for the 600 Hz ANF, stim @ BF
xlabel('Intensity');
ylabel('Rate');
title('Rate vs Intensity');
legend('ah','600Hz');


%% Recording for all ANFs at 3 dB levels
% taking the 3 intensities as specified in the first problem statement
fwTime = length(fivewo)/Fs;
rate_0 = zeros(48,fwTime*Fs*2); % Adjusted for 48 fibers as per the first problem statement
rate_35 = zeros(48,fwTime*Fs*2);
rate_70 = zeros(48,fwTime*Fs*2);

% playing fivewo for the ANFs
for i = 1:48 % Adjusted for 48 fibers as per the first problem statement
    CF = ANF(i);
    Input = fivewo .* 10^(-70/20); % Adjusted intensity for -70 dB SPL
    vihc = catmodel_IHC(Input,CF,nrep,1/Fs,fwTime*2,cohc,cihc); 
    [synout,psth] = catmodel_Synapse(vihc,CF,nrep,1/Fs,fiberType,implnt); 
    rate_0(i,:) = psth;
end

for i = 1:48 % Adjusted for 48 fibers as per the first problem statement
    CF = ANF(i);
    Input = fivewo .* 10^(-40/20); % Adjusted intensity for -40 dB SPL
    vihc = catmodel_IHC(Input,CF,nrep,1/Fs,fwTime*2,cohc,cihc); 
    [synout,psth] = catmodel_Synapse(vihc,CF,nrep,1/Fs,fiberType,implnt); 
    rate_35(i,:) = psth;
end

for i = 1:48 % Adjusted for 48 fibers as per the first problem statement
    CF = ANF(i);
    Input = fivewo .* 10^(-5/20); % Adjusted intensity for -5 dB SPL
    vihc = catmodel_IHC(Input,CF,nrep,1/Fs,fwTime*2,cohc,cihc); 
    [synout,psth] = catmodel_Synapse(vihc,CF,nrep,1/Fs,fiberType,implnt); 
    rate_70(i,:) = psth;
end

%% Plotting the actual spectrogram and the average ANF response rates
fig = figure;
fig.Position(3) = 1000; fig.Position(4) = 600;
spectrogram(fivewo, hann(25.6e-3*Fs), 12.8e-3*Fs, 1:8000, Fs, 'yaxis'); set(gca, 'yscale', 'log');
title('Spectrogram for the speech signal');
p = (2:7);
wind = zeros(1,6);
for i = 1:length(p)
    wind(i) = 1e-3 * Fs * (2^p(i));
end

% wind(i) has the number of samples in each window
winShift = floor(wind/2);

fig = figure;
fig.Position(3) = 1000; fig.Position(4) = 600;
F = ANF(1:48); % Adjusted for 48 fibers as per the first problem statement
for w = 1:6      % for each window size
    t2 = wind(w)/2 : winShift(w) : length(fivewo)-wind(w)/2;
    avg_rates = zeros(length(F), length(t2));
    for f = 1:48 % Adjusted for 48 fibers as per the first problem statement
        for i = 1:length(t2) % b for bin number
            xo = rate_70(f,(t2(i)-winShift(w)+1):(t2(i)+winShift(w)));
            avg_rates(f,i) = sum(xo)*Fs / wind(w);
        end
    end
    subplot(2,3,w);
    [tim, frq ] = meshgrid(t2, F);
    surf(tim, frq, avg_rates/nrep,'edgecolor','none');
    %set(gca,'xtick',[]);set(gca,'ytick',[]);
    set(gca, 'yscale', 'log');%xlabel([]);ylabel([]);
    xlim([0,1.5e5]);
    title(['Window Size = ',num2str(wind(w)/(1e-3*Fs)),'ms']);
    xlabel('Time');
    ylabel('Frequency');
    view(2);
end

%% Question-3:

fig = figure; fig.Position(3) = 1200; fig.Position(4) = 800;
spectrogram(fivewo, hann(12.8e-3*Fs), 6.4e-3*Fs, 1:8000, Fs, 'yaxis'); %view(3);

nF =  [6, 10, 14, 18, 22, 26, 30, 34, 38];
cmap1 = hsv(9);
win = 12.8e-3*Fs;
wshift = floor(win/2);
t3 = win/2 : wshift : length(fivewo)-win/2;
fre_pt = zeros(1,length(t3));

fig = figure; fig.Position(3) = 1600; fig.Position(4) = 800;

% Plot spectrogram first
spectrogram(fivewo, hann(12.8e-3*Fs), 6.4e-3*Fs, 1:8000, Fs, 'yaxis');
hold on;
for f = 1 : 9
    for i = 6 : length(t3)
        Xp = rate_70(nF(f), (t3(i)-wshift+1) : (t3(i)+wshift));
        m = mean(Xp);
        FFT = abs(fft(Xp - m));
        [M,I] = max(squeeze(FFT(1:length(FFT)/2)));
        fre_pt(i) = I*Fs/length(FFT);
    end
    scatter3(t3/Fs,fre_pt/1000,ones(1,length(t3))*10,[],cmap1(f,:),'filled', 'MarkerEdgeColor', 'k');
    ylim([0 3]); hold on; 
end

view(2);


%% Part:B (Extra Credit)
% Load audio file information and data
info = audioinfo('fivewo.wav');
[x, Fs] = audioread('fivewo.wav');
% Create a time vector for plotting
t = 0:seconds(1/Fs):seconds(info.Duration);
t = t(1:end-1);
% Plot the original audio waveform
subplot(3, 1, 1);
plot(t, x);
xlabel('Time');
ylabel('x(t)');
title('Plot of the Given Audio');
% Perform FFT on the audio signal
N = 800;
y_ = fft(x, N);
y_ = fftshift(y_);
M = abs(y_);
M = M/N;
f = (-length(y_)/2:length(y_)/2-1) * Fs/length(y_);
subplot(3, 1, 2);
plot(f, M);
xlabel('Frequency')
ylabel('X(f)')
title('FFT of Audio Signal')
% Generate white Gaussian noise with a specified SNR
n = 100; % SNR
noise = (1/n) * wgn(156250, 1, 1);
subplot(3, 1, 3);
plot(t, noise);
xlabel('Time');
ylabel('Noise');
title('Plot of the White Gaussian Noise');
% Initialize the result variable z
z = 0;
% Define the number of bandpass filters to apply
N = [1, 2, 4, 8];
fig = figure; fig.Position(3) = 2500; fig.Position(4) = 1700;

% Loop through different numbers of bandpass filters
for j = 1:length(N)
 k = N(j);
 for i = 1:k
 % Define bandpass filter parameters
 f1 = 250 * 8^((i-1)/k);
 f2 = 250 * 8^(i/k);
 
 % Design the bandpass filter
 [B, A] = butter(2, [f1/Fs, f2/Fs], 'stop');
 
 % Apply the bandpass filter to the audio
 y = filter(B, A, x);
 
 % Compute the Hilbert transform and apply a low-pass filter
 y_hilb = hilbert(y);
 [B_, A_] = butter(2, 240/Fs, 'low');
 y_final = filter(B_, A_, abs(y_hilb));
 
 % Multiply the filtered audio by the noise
 mult = y_final .* noise;
 
 % Accumulate the results
 z = z + mult;
 end
 
 % Plot the final audio with a specific number of bandpass filters
 subplot(2, 2, j);
 plot(t, z);
 xlabel('Time');
 ylabel('z(t)');
 title(['Plot of Final Audio with number of bandpass filters: ',num2str(k)]);
 
 % Save the result as an output audio file
 filename = ['output', num2str(j), '.wav'];
 audiowrite(filename, z, Fs)
end


%% Audio with 1 band
%% Separating out 'ah', generating different intensity stimuli 

h = (0:1/8:6); % Adjusted to span 6 octaves
ANF = zeros(1,length(h));
for k = 1:length(h)
    ANF(k) = 125*(2^h(k)); % Adjusted starting frequency to 125 Hz
end

intensities = -20:5:80; % Intensities specified in the first problem statement
[fivewo, fs] = audioread('output1.wav');
fivewo = fivewo';
S = 1+round(1.05*fs);
E = 1+round(1.15*fs);
L = length(fivewo);
[y,fs] = audioread('output1.wav',[1+round(1.05*fs),1+round(1.15*fs)]);
y = y';
Fs = 100e3;
rt = 10e-3;
%sound(y,Fs);
tester = [fivewo(1:S) zeros(1,50000) fivewo(S:E) zeros(1,50000) fivewo(E:L)] ;
sound(tester,Fs);
%t = 0:1/Fs:0.2*fs/Fs; % time vector
t = (0:length(y)-1)/Fs;
mxpts = length(t);
irpts = rt*Fs;
y(1:irpts) = y(1:irpts) .* ((0:(irpts-1))/irpts); 
y((mxpts-irpts):mxpts) = y((mxpts-irpts):mxpts).*((irpts:-1:0)/irpts);
%s = size(t);
x = rms(y);
I_rms = 20*log10(x/(20*10^(-6)));
rate_matrix2 = zeros(1,21);
Input = zeros(1,length(y));
cohc  = 1.0;   % normal ohc function
cihc  = 1.0;   % normal ihc function
fiberType = 3;
implnt = 0;
nrep = 80; % Adjusted repetitions as per the first problem statement
CF = 600; % Adjusted to 600 Hz as per the first problem statement

psthbinwidth = 0.5e-3;
%psthbins = round(psthbinwidth*Fs);  % number of psth bins per psth bin
ahTime = length(y) / Fs ;
for i = 1:21 
    Input = y * 10^(((i-1)*10 - 95)/20);
    vihc = catmodel_IHC(Input,CF,nrep,1/Fs,ahTime*2,cohc,cihc); 
    [synout,psth] = catmodel_Synapse(vihc,CF,nrep,1/Fs,fiberType,implnt); 
    %{
    timeout = (1:length(psth))*1/Fs;
    psthtime = timeout(1:psthbins:end); % time vector for psth
    pr = sum(reshape(psth,psthbins,length(psth)/psthbins))/nrep; % pr of spike in each bin
    Psth = pr/psthbinwidth; % psth in units of spikes/s
    %}
    rate_matrix2(i) = sum(psth(1:(length(psth)/2)));
end

rate_matrix2 = rate_matrix2 / nrep;
rate_matrix2 = rate_matrix2 / ahTime ;


%% Plotting Rate vs Intensity of 'ah' and that of 600 Hz ANF for stimulus at BF

fig = figure;
grid on;
hold on;
plot((-20:5:80), rate_matrix2);         % for the ah stimulus
plot((-20:5:80), rate_matrix(19,:));   % for the 600 Hz ANF, stim @ BF 
xlabel('Intensity'); ylabel('Rate'); title('Rate vs Intensity'); legend('ah','600Hz');

%% Recording for all ANFs at 3 dB levels
% taking the 3 intensities as specified in the first problem statement
fwTime = length(fivewo)/Fs;
rate_0 = zeros(48,fwTime*Fs*2); % Adjusted for 48 fibers as per the first problem statement
rate_35 = zeros(48,fwTime*Fs*2);
rate_70 = zeros(48,fwTime*Fs*2);

% playing fivewo for the ANFs
for i = 1:48 % Adjusted for 48 fibers as per the first problem statement
    CF = ANF(i);
    Input = fivewo .* 10^(-70/20); % Adjusted intensity for -70 dB SPL
    vihc = catmodel_IHC(Input,CF,nrep,1/Fs,fwTime*2,cohc,cihc); 
    [synout,psth] = catmodel_Synapse(vihc,CF,nrep,1/Fs,fiberType,implnt); 
    rate_0(i,:) = psth;
end
for i = 1:48 % Adjusted for 48 fibers as per the first problem statement
    CF = ANF(i);
    Input = fivewo .* 10^(-40/20); % Adjusted intensity for -40 dB SPL
    vihc = catmodel_IHC(Input,CF,nrep,1/Fs,fwTime*2,cohc,cihc); 
    [synout,psth] = catmodel_Synapse(vihc,CF,nrep,1/Fs,fiberType,implnt); 
    rate_35(i,:) = psth;
end
for i = 1:48 % Adjusted for 48 fibers as per the first problem statement
    CF = ANF(i);
    Input = fivewo .* 10^(-5/20); % Adjusted intensity for -5 dB SPL
    vihc = catmodel_IHC(Input,CF,nrep,1/Fs,fwTime*2,cohc,cihc); 
    [synout,psth] = catmodel_Synapse(vihc,CF,nrep,1/Fs,fiberType,implnt); 
    rate_70(i,:) = psth;
end

 p = (2:7);
wind = zeros(1,6);
for i = 1:length(p)
    wind(i) = 1e-3 * Fs * (2^p(i));
end

% wind(i) has the number of samples in each window
winShift = floor(wind/2);

fig = figure;
fig.Position(3) = 1000; fig.Position(4) = 600;
F = ANF(1:48); % Adjusted for 48 fibers as per the first problem statement
for w = 1:6      % for each window size
    t2 = wind(w)/2 : winShift(w) : length(fivewo)-wind(w)/2;
    avg_rates = zeros(length(F), length(t2));
    for f = 1:48 % Adjusted for 48 fibers as per the first problem statement
        for i = 1:length(t2) % b for bin number
            xo = rate_70(f,(t2(i)-winShift(w)+1):(t2(i)+winShift(w)));
            avg_rates(f,i) = sum(xo)*Fs / wind(w);
        end
    end
    subplot(2,3,w);
    [tim, frq ] = meshgrid(t2, F);
    surf(tim, frq, avg_rates/nrep,'edgecolor','none');
    %set(gca,'xtick',[]);set(gca,'ytick',[]);
    set(gca, 'yscale', 'log');%xlabel([]);ylabel([]);
    xlim([0,1.5e5]);
    title(['Window Size = ',num2str(wind(w)/(1e-3*Fs)),'ms']);
    xlabel('Time');
    ylabel('Frequency');
    view(2);
end

%% Part-B: 8 band audio to A3 spectrogram

nF =  [6, 10, 14, 18, 22, 26, 30, 34, 38];
cmap1 = hsv(9);
win = 12.8e-3*Fs;
wshift = floor(win/2);
t3 = win/2 : wshift : length(fivewo)-win/2;
fre_pt = zeros(1,length(t3));

fig = figure; fig.Position(3) = 1600; fig.Position(4) = 800;

% Plot spectrogram first
spectrogram(fivewo, hann(12.8e-3*Fs), 6.4e-3*Fs, 1:8000, Fs, 'yaxis');
hold on;

for f = 1 : 9
    for i = 6 : length(t3)
        Xp = rate_70(nF(f), (t3(i)-wshift+1) : (t3(i)+wshift));
        m = mean(Xp);
        FFT = abs(fft(Xp - m));
        [M,I] = max(squeeze(FFT(1:length(FFT)/2)));
        fre_pt(i) = I*Fs/length(FFT);
    end
    scatter3(t3/Fs,fre_pt/1000,ones(1,length(t3))*10,[],cmap1(f,:),'filled', 'MarkerEdgeColor', 'k');
    ylim([0 3]); hold on; 
end

view(2);

%% 8-Band Audio Processing through A2 and A3

%% Question-2
%% Separating out 'ah', generating different intensity stimuli 

h = (0:1/8:6); % Adjusted to span 6 octaves
ANF = zeros(1,length(h));
for k = 1:length(h)
    ANF(k) = 125*(2^h(k)); % Adjusted starting frequency to 125 Hz
end

intensities = -20:5:80; % Intensities specified in the first problem statement
[fivewo, fs] = audioread('output4.wav');
fivewo = fivewo';
S = 1+round(1.05*fs);
E = 1+round(1.15*fs);
L = length(fivewo);
[y,fs] = audioread('output4.wav',[1+round(1.05*fs),1+round(1.15*fs)]);
y = y';
Fs = 100e3;
rt = 10e-3;
%sound(y,Fs);
tester = [fivewo(1:S) zeros(1,50000) fivewo(S:E) zeros(1,50000) fivewo(E:L)] ;
sound(tester,Fs);
%t = 0:1/Fs:0.2*fs/Fs; % time vector
t = (0:length(y)-1)/Fs;
mxpts = length(t);
irpts = rt*Fs;
y(1:irpts) = y(1:irpts) .* ((0:(irpts-1))/irpts); 
y((mxpts-irpts):mxpts) = y((mxpts-irpts):mxpts).*((irpts:-1:0)/irpts);
%s = size(t);
x = rms(y);
I_rms = 20*log10(x/(20*10^(-6)));
rate_matrix2 = zeros(1,21);
Input = zeros(1,length(y));
cohc  = 1.0;   % normal ohc function
cihc  = 1.0;   % normal ihc function
fiberType = 3;
implnt = 0;
nrep = 80; % Adjusted repetitions as per the first problem statement
CF = 600; % Adjusted to 600 Hz as per the first problem statement

psthbinwidth = 0.5e-3;
%psthbins = round(psthbinwidth*Fs);  % number of psth bins per psth bin
ahTime = length(y) / Fs ;
for i = 1:21 
    Input = y * 10^(((i-1)*10 - 95)/20);
    vihc = catmodel_IHC(Input,CF,nrep,1/Fs,ahTime*2,cohc,cihc); 
    [synout,psth] = catmodel_Synapse(vihc,CF,nrep,1/Fs,fiberType,implnt); 
    %{
    timeout = (1:length(psth))*1/Fs;
    psthtime = timeout(1:psthbins:end); % time vector for psth
    pr = sum(reshape(psth,psthbins,length(psth)/psthbins))/nrep; % pr of spike in each bin
    Psth = pr/psthbinwidth; % psth in units of spikes/s
    %}
    rate_matrix2(i) = sum(psth(1:(length(psth)/2)));
end

rate_matrix2 = rate_matrix2 / nrep;
rate_matrix2 = rate_matrix2 / ahTime ;


%% Plotting Rate vs Intensity of 'ah' and that of 600 Hz ANF for stimulus at BF

fig = figure;
grid on;
hold on;
plot((-20:5:80), rate_matrix2);         % for the ah stimulus
plot((-20:5:80), rate_matrix(19,:));   % for the 600 Hz ANF, stim @ BF
xlabel('Intensity');
ylabel('Rate');
title('Rate vs Intensity');
legend('ah','600Hz');

%% Recording for all ANFs at 3 dB levels
% taking the 3 intensities as specified in the first problem statement
fwTime = length(fivewo)/Fs;
rate_0 = zeros(48,fwTime*Fs*2); % Adjusted for 48 fibers as per the first problem statement
rate_35 = zeros(48,fwTime*Fs*2);
rate_70 = zeros(48,fwTime*Fs*2);

% playing fivewo for the ANFs
for i = 1:48 % Adjusted for 48 fibers as per the first problem statement
    CF = ANF(i);
    Input = fivewo .* 10^(-70/20); % Adjusted intensity for -70 dB SPL
    vihc = catmodel_IHC(Input,CF,nrep,1/Fs,fwTime*2,cohc,cihc); 
    [synout,psth] = catmodel_Synapse(vihc,CF,nrep,1/Fs,fiberType,implnt); 
    rate_0(i,:) = psth;
end
for i = 1:48 % Adjusted for 48 fibers as per the first problem statement
    CF = ANF(i);
    Input = fivewo .* 10^(-40/20); % Adjusted intensity for -40 dB SPL
    vihc = catmodel_IHC(Input,CF,nrep,1/Fs,fwTime*2,cohc,cihc); 
    [synout,psth] = catmodel_Synapse(vihc,CF,nrep,1/Fs,fiberType,implnt); 
    rate_35(i,:) = psth;
end
for i = 1:48 % Adjusted for 48 fibers as per the first problem statement
    CF = ANF(i);
    Input = fivewo .* 10^(-5/20); % Adjusted intensity for -5 dB SPL
    vihc = catmodel_IHC(Input,CF,nrep,1/Fs,fwTime*2,cohc,cihc); 
    [synout,psth] = catmodel_Synapse(vihc,CF,nrep,1/Fs,fiberType,implnt); 
    rate_70(i,:) = psth;
end

p = (2:7);
wind = zeros(1,6);
for i = 1:length(p)
    wind(i) = 1e-3 * Fs * (2^p(i));
end

% wind(i) has the number of samples in each window
winShift = floor(wind/2);

fig = figure;
fig.Position(3) = 1000; fig.Position(4) = 600;
F = ANF(1:48); % Adjusted for 48 fibers as per the first problem statement
for w = 1:6      % for each window size
    t2 = wind(w)/2 : winShift(w) : length(fivewo)-wind(w)/2; 
    avg_rates = zeros(length(F), length(t2));
    for f = 1:48 % Adjusted for 48 fibers as per the first problem statement
        for i = 1:length(t2) % b for bin number
            xo = rate_70(f,(t2(i)-winShift(w)+1):(t2(i)+winShift(w)));
            avg_rates(f,i) = sum(xo)*Fs / wind(w);
        end
    end
    subplot(2,3,w);
    [tim, frq ] = meshgrid(t2, F);
    surf(tim, frq, avg_rates/nrep,'edgecolor','none');
    %set(gca,'xtick',[]);set(gca,'ytick',[]);
    set(gca, 'yscale', 'log');%xlabel([]);ylabel([]);
    xlim([0,1.5e5]);
    title(['Window Size = ',num2str(wind(w)/(1e-3*Fs)),'ms']);
    xlabel('Time');
    ylabel('Frequency');
    view(2);
end

%% Part-B: 8 band audio file to Ques. A3 spectrogram

nF =  [6, 10, 14, 18, 22, 26, 30, 34, 38];
cmap1 = hsv(9);
win = 12.8e-3*Fs;
wshift = floor(win/2);
t3 = win/2 : wshift : length(fivewo)-win/2;
fre_pt = zeros(1,length(t3));

fig = figure; fig.Position(3) = 1600; fig.Position(4) = 800;

% Plot spectrogram first
spectrogram(fivewo, hann(12.8e-3*Fs), 6.4e-3*Fs, 1:8000, Fs, 'yaxis');
hold on;

for f = 1 : 9
    for i = 6 : length(t3)
        Xp = rate_70(nF(f), (t3(i)-wshift+1) : (t3(i)+wshift));
        m = mean(Xp);
        FFT = abs(fft(Xp - m));
        [M,I] = max(squeeze(FFT(1:length(FFT)/2)));
        fre_pt(i) = I*Fs/length(FFT);
    end
    scatter3(t3/Fs,fre_pt/1000,ones(1,length(t3))*10,[],cmap1(f,:),'filled', 'MarkerEdgeColor', 'k');
    ylim([0 3]); hold on; 
end

view(2);
