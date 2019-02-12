function imOut = radUnDist(imIn, k1, k2, nSteps)
    % Your code here
    [m, n] = size(imIn);
    [ax, ay]=meshgrid(1:n, 1:m);

    cx = m/2;
    cy = n/2;

    ax_orig = (ax - cx)./cx;
    ay_orig = (ay - cy)./cy;

    x= ax_orig;
    y= ay_orig;

    for i=1:nSteps
        r2 = sqrt(x.^2 + y.^2);
        dr = - k1*r2 - k2*r2.^2;   
        x =  ax_orig + x.*dr;
        y =  ay_orig + y.*dr;
    end     
    imOut = interp2(imIn, x*cx + cx, y*cy + cy, 'cubic');
end