j=20000;
step=1000;
windowlength=10;
[data,fs]=audioread('clean.wav');
for j=16000:step:500000;
    i=j/step;
    A(:,i)=data(j:j+20*fs*10^-3);
    
    subplot(1,3,1);
    plot([]);
    plot(A(:,i));
    sound(A(:,i),FS);
    subplot(1,3,2);
%     hold on;
    b = ones(1,windowlength)/windowlength;
    plot(abs(filtfilt(b,1,fft(A(:,i)))));
    subplot(1,3,3);
%     hold on;
    plot(bart(A(:,i),10));
    
    k = waitforbuttonpress ;
end



