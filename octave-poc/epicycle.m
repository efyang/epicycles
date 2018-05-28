clear
%x = [-5:5] + 1i;
%x = [-2+2*i, -1+2*i, 2*i, 1+2*i, 2+2*i, 2+i, 2, 2-i, 2-2*i, 1-2*i, -2*i, -1-2*i, -2-2*i, -2-i, -2, -2+i];
%x = 3*[-2+2*i, -1+2*i, 2*i, 1+2*i, 2+2*i, 2+i, 2, 2-i, 2-2*i, 1-2*i, -2*i, -1-2*i, -2-2*i, -2-i, -2, -2+i, -2+2*i];
x = [0:0.5:5] * (1 + i);
plot(x, 'b')
set(gca,'XLim',[-10 10], 'YLim',[-10 10]);
y = fftshift(fft(x));
n = length(y);
radii = abs(y)/n;
phase_angles = atan2(imag(y), real(y));
points = [];
P = 100;
for t = [0:P]
  curpt = 0 + 0i;
  plot(x, 'g')
  set(gca,'XLim',[-10 10], 'YLim',[-10 10]);
  hold on;
  % plot each circle
  for j = 1:n
    r = radii(j);
    phase_angle = phase_angles(j);
    circle(real(curpt), imag(curpt), r);
    offset = y(j)/n .* exp(i * 2 * pi * (j - ceil(n/2) - mod(n+1,2) ) * (t/P));
    % this also works
    % offset = r .* exp(i * ( 2 * pi * (j - ceil(n/2)) * (t/P) + phase_angle));
    next_center = curpt + offset;
    % plot the radius
    plot([real(curpt), real(next_center)], [imag(curpt), imag(next_center)], 'k');
    curpt = next_center;
  endfor
  points = [points curpt];
  plot(points, 'r');
  drawnow;
  pause(1/30);
  hold off;
endfor
%plot(points);
%plot(ifft(y), 'g');
hold off