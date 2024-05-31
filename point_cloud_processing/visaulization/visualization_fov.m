% Define cone parameters
d = 6;
r = 0.61 * d;  % Radius
h = 1 * d;     % Height
n = 30;        % Number of triangular facets

% Create cone vertices
theta = linspace(0, 2*pi, n);
x0 = r * cos(theta);
y0 = r * sin(theta);
z0 = ones(size(x0)) * h;  % All z-coordinates set to height

% Add cone apex
x0 = [x0, 0];
y0 = [y0, 0];
z0 = [z0, 0];

% Create triangular facets
triangles0 = [];
for i = 1:(n-1)
    triangles0 = [triangles0; [i, i+1, n+1]];
end
triangles0 = [triangles0; [n, 1, n+1]];

% Define ellipse parameters
a = 1.05 * d;  % x-axis radius
b = 0.26 * d;  % y-axis radius

% Create ellipse vertices
x1 = a * cos(theta);
y1 = b * sin(theta);
z1 = ones(size(x1)) * h;  % All z-coordinates set to height

% Add ellipse center
x1 = [x1, 0];
y1 = [y1, 0];
z1 = [z1, 0];

% Create triangular facets for ellipse
triangles1 = triangles0;

% Define cone parameters for lidar
r = 20; % Radius
h = 6; % Height

% Create cone vertices for lidar
theta = linspace(0, 2*pi, n);
x2 = r * cos(theta);
y2 = r * sin(theta);
z2 = ones(size(x2)) * h;  % All z-coordinates set to height

% Add cone apex for lidar
x2 = [x2, 0];
y2 = [y2, 0];
z2 = [z2, 0];

% Create triangular facets for lidar
triangles2 = triangles0;

% Define sphere parameters
radius = 40;

% Generate sphere data
theta = linspace(0, 2*pi, 100);
phi = linspace(0, pi, 100);
[theta, phi] = meshgrid(theta, phi);
x = radius * sin(phi) .* cos(theta);
y = radius * sin(phi) .* sin(theta);
z = radius * cos(phi);

% Set height threshold
h0 = 6;

% Keep data below height threshold
x = x(z > 0 & z < h0);
y = y(z > 0 & z < h0);
z = z(z > 0 & z < h0);

% Plot cones and sphere
figure;
hold on;
trisurf(triangles0, x0, y0, z0, 'FaceColor', 'r', 'FaceAlpha', 0.2, 'EdgeAlpha', 0);
trisurf(triangles1, x1, y1, z1, 'FaceColor', 'b', 'FaceAlpha', 0.2, 'EdgeAlpha', 0);
trisurf(triangles2, x2, y2, z2, 'FaceColor', 'g', 'FaceAlpha', 0.2, 'EdgeAlpha', 0);
plot3(x(:), y(:), z(:), 'g', 'LineWidth', 1);

axis equal;
xlabel('X');
ylabel('Y');
zlabel('Z');
view(3);  % Set view direction
xlim([-4, 4]);  % Adjust limits if necessary
ylim([-4, 4]);
zlim([0, 6]);
hold off;
