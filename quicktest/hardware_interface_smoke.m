project_root = fileparts(fileparts(mfilename('fullpath')));
addpath(fullfile(project_root, 'backend', 'interface', 'OSA'));
addpath(fullfile(project_root, 'backend', 'interface', 'VoltageSource'));

had_error = false;

fprintf('BEGIN_OSA\n');
try
    osa_result = wavelength_sweep( ...
        'lam_start_nm', 1549.5, ...
        'lam_stop_nm', 1550.5, ...
        'resolution_nm', 0.1, ...
        'points_per_resolution', 5, ...
        'speed', '2x', ...
        'board_index', 0, ...
        'primary_address', 1, ...
        'timeout_s', 30, ...
        'plot_result', false, ...
        'record', false ...
    );
    fprintf('OSA_POINTS=%d\n', numel(osa_result.wavelength_nm));
    fprintf('OSA_START_NM=%.6f\n', osa_result.wavelength_nm(1));
    fprintf('OSA_STOP_NM=%.6f\n', osa_result.wavelength_nm(end));
    fprintf('OSA_MAX_DBM=%.6f\n', max(osa_result.power_dbm));
    fprintf('OSA_MIN_DBM=%.6f\n', min(osa_result.power_dbm));
catch ME
    had_error = true;
    fprintf(2, 'OSA_ERROR_ID=%s\n', ME.identifier);
    fprintf(2, 'OSA_ERROR_MSG=%s\n', strrep(ME.message, newline, ' '));
end

channels = 1:12;
drive_voltage = 0.1;

fprintf('BEGIN_VOLTAGE_SOURCE\n');
try
    silicon_extreme_api('connect', 7);
    voltage_cleanup = onCleanup(@() local_voltage_cleanup(channels)); %#ok<NASGU>

    initial_snapshot = silicon_extreme_api('snapshot', channels);
    fprintf('VS_INITIAL_V=%s\n', mat2str(initial_snapshot.voltages, 6));
    fprintf('VS_INITIAL_I=%s\n', mat2str(initial_snapshot.currents, 6));

    silicon_extreme_api('configure_limits', channels, 1.0, []);
    silicon_extreme_api('set_voltages', channels, drive_voltage * ones(size(channels)));
    pause(0.5);

    driven_snapshot = silicon_extreme_api('snapshot', channels);
    fprintf('VS_DRIVEN_V=%s\n', mat2str(driven_snapshot.voltages, 6));
    fprintf('VS_DRIVEN_I=%s\n', mat2str(driven_snapshot.currents, 6));
    fprintf('VS_NONZERO_CURRENT_COUNT=%d\n', nnz(abs(driven_snapshot.currents) > 0));

    silicon_extreme_api('set_voltages', channels, zeros(size(channels)));
    pause(0.5);

    reset_snapshot = silicon_extreme_api('snapshot', channels);
    fprintf('VS_RESET_V=%s\n', mat2str(reset_snapshot.voltages, 6));
    fprintf('VS_RESET_I=%s\n', mat2str(reset_snapshot.currents, 6));

    silicon_extreme_api('disconnect');
catch ME
    had_error = true;
    fprintf(2, 'VS_ERROR_ID=%s\n', ME.identifier);
    fprintf(2, 'VS_ERROR_MSG=%s\n', strrep(ME.message, newline, ' '));
    try
        silicon_extreme_api('set_voltages', channels, zeros(size(channels)));
    catch
    end
    try
        silicon_extreme_api('disconnect');
    catch
    end
end

if had_error
    error('hardware_interface_smoke:failed', 'One or more hardware checks failed.');
end

function local_voltage_cleanup(channels)
try
    silicon_extreme_api('set_voltages', channels, zeros(size(channels)));
catch
end

try
    silicon_extreme_api('disconnect');
catch
end
end
