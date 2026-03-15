function result = wavelength_sweep(varargin)
%WAVELENGTH_SWEEP Acquire spectrum data from the OSA through GPIB.
%   RESULT = WAVELENGTH_SWEEP(Name, Value, ...) configures the OSA, performs
%   a single sweep, and returns a struct with the measured spectrum.
%
%   Name-value parameters:
%       lam_start_nm                  double scalar, default 1548
%       lam_stop_nm                   double scalar, default 1552
%       resolution_nm                 double scalar, default 0.02
%       sensitivity                   char/string, default 'high2'
%       points_per_resolution         integer scalar, default 5
%       speed                         char/string, default '2x'
%       reflevel_up_dbm               double scalar, default 0
%       reflevel_down_dbm             double scalar, default -100
%       channel                       char/string, default 'a'
%       board_index                   integer scalar, default 0
%       primary_address               integer scalar, default 1
%       timeout_s                     double scalar, default 150
%       input_buffer_size             integer scalar, default 540027
%       output_buffer_size            integer scalar, default 540027
%       use_requested_range           logical/numeric scalar, default true
%       restore_defaults              logical/numeric scalar, default true
%       record                        logical/numeric scalar, default false
%       save_path                     char/string, default ''
%       plot_result                   logical/numeric scalar, default nargout == 0
%       normalization                 logical/numeric scalar, default false
%       normalization_reference_path  char/string, default ''
%       target_wavelengths_nm         numeric vector, default []
%       interpolation_method          char/string, default 'linear'
%
%   RESULT fields:
%       wavelength_nm
%       raw_power_dbm
%       power_dbm
%       data
%       selected_wavelength_nm
%       selected_power_dbm
%       settings

    parser = inputParser;
    parser.FunctionName = 'wavelength_sweep';

    addParameter(parser, 'lam_start_nm', 1548, @localNumericScalar);
    addParameter(parser, 'lam_stop_nm', 1552, @localNumericScalar);
    addParameter(parser, 'resolution_nm', 0.02, @localPositiveScalar);
    addParameter(parser, 'sensitivity', 'high2', @localTextScalar);
    addParameter(parser, 'points_per_resolution', 5, @localPositiveIntegerScalar);
    addParameter(parser, 'speed', '2x', @localTextScalar);
    addParameter(parser, 'reflevel_up_dbm', 0, @localNumericScalar);
    addParameter(parser, 'reflevel_down_dbm', -100, @localNumericScalar);
    addParameter(parser, 'channel', 'a', @localTextScalar);
    addParameter(parser, 'board_index', 0, @localNonNegativeIntegerScalar);
    addParameter(parser, 'primary_address', 1, @localPositiveIntegerScalar);
    addParameter(parser, 'timeout_s', 150, @localPositiveScalar);
    addParameter(parser, 'input_buffer_size', 180009 * 3, @localPositiveIntegerScalar);
    addParameter(parser, 'output_buffer_size', 180009 * 3, @localPositiveIntegerScalar);
    addParameter(parser, 'use_requested_range', true, @localLogicalScalar);
    addParameter(parser, 'restore_defaults', true, @localLogicalScalar);
    addParameter(parser, 'record', false, @localLogicalScalar);
    addParameter(parser, 'save_path', '', @localTextScalar);
    addParameter(parser, 'plot_result', nargout == 0, @localLogicalScalar);
    addParameter(parser, 'normalization', false, @localLogicalScalar);
    addParameter(parser, 'normalization_reference_path', '', @localTextScalar);
    addParameter(parser, 'target_wavelengths_nm', [], @localNumericVector);
    addParameter(parser, 'interpolation_method', 'linear', @localTextScalar);

    parse(parser, varargin{:});
    settings = parser.Results;

    if settings.lam_stop_nm <= settings.lam_start_nm
        error('wavelength_sweep:InvalidRange', ...
            'lam_stop_nm must be greater than lam_start_nm.');
    end

    settings.channel = lower(char(settings.channel));
    settings.sensitivity = char(settings.sensitivity);
    settings.speed = char(settings.speed);
    settings.save_path = char(settings.save_path);
    settings.normalization_reference_path = char(settings.normalization_reference_path);
    settings.interpolation_method = char(settings.interpolation_method);
    settings.target_wavelengths_nm = reshape(double(settings.target_wavelengths_nm), 1, []);
    settings.point_count = round((settings.lam_stop_nm - settings.lam_start_nm) ...
        * settings.points_per_resolution / settings.resolution_nm) + 1;

    osa = localGetGpibObject(settings.board_index, settings.primary_address);
    cleanup = onCleanup(@() localCleanupOsa(osa, settings)); %#ok<NASGU>

    set(osa, 'InputBufferSize', settings.input_buffer_size);
    set(osa, 'OutputBufferSize', settings.output_buffer_size);
    set(osa, 'Timeout', settings.timeout_s);

    fopen(osa);
    localConfigureSweep(osa, settings);
    localWaitForSweep(osa, settings.timeout_s);

    fprintf(osa, ':format:data ascii');
    fprintf(osa, [':trac:y? tr', settings.channel]);
    powerText = fscanf(osa);
    fprintf(osa, [':trac:x? tr', settings.channel]);
    wavelengthText = fscanf(osa);

    rawPowerDbm = localParseAsciiTrace(powerText);
    wavelengthNm = localParseAsciiTrace(wavelengthText) * 1e9;

    if numel(rawPowerDbm) ~= numel(wavelengthNm)
        error('wavelength_sweep:TraceLengthMismatch', ...
            'The OSA returned %d power points and %d wavelength points.', ...
            numel(rawPowerDbm), numel(wavelengthNm));
    end

    if numel(rawPowerDbm) ~= settings.point_count
        warning('wavelength_sweep:UnexpectedPointCount', ...
            'Expected %d points from the OSA but received %d. Using the instrument data as-is.', ...
            settings.point_count, numel(rawPowerDbm));
    end

    reportedPowerDbm = rawPowerDbm;

    if settings.normalization
        if isempty(settings.normalization_reference_path)
            error('wavelength_sweep:MissingNormalizationReference', ...
                'normalization_reference_path is required when normalization is enabled.');
        end

        referenceData = load(settings.normalization_reference_path);
        if size(referenceData, 2) < 2
            error('wavelength_sweep:InvalidNormalizationReference', ...
                'Normalization reference must have at least two columns.');
        end

        referencePowerDbm = interp1(referenceData(:, 1), referenceData(:, 2), ...
            wavelengthNm, settings.interpolation_method, NaN);
        reportedPowerDbm = rawPowerDbm - referencePowerDbm;
    end

    data = [wavelengthNm(:), reportedPowerDbm(:)];

    if settings.record
        if isempty(settings.save_path)
            error('wavelength_sweep:MissingSavePath', ...
                'save_path is required when record is enabled.');
        end
        save(settings.save_path, '-ascii', 'data');
    end

    selectedWavelengthNm = [];
    selectedPowerDbm = [];
    if ~isempty(settings.target_wavelengths_nm)
        selectedWavelengthNm = settings.target_wavelengths_nm(:).';
        selectedPowerDbm = interp1(wavelengthNm, reportedPowerDbm, ...
            selectedWavelengthNm, settings.interpolation_method, NaN);
    end

    if settings.plot_result
        figure(1);
        clf;
        set(gcf, 'unit', 'centimeters', 'position', [5, 5, 15, 10]);
        plot(wavelengthNm, reportedPowerDbm, 'linewidth', 2);
        grid on;
        axis([settings.lam_start_nm, settings.lam_stop_nm, ...
            settings.reflevel_down_dbm, settings.reflevel_up_dbm]);
        xlabel('Wavelength (nm)');
        ylabel('Power (dBm)');
        title('OSA Spectrum');
    end

    result = struct( ...
        'wavelength_nm', wavelengthNm(:).', ...
        'raw_power_dbm', rawPowerDbm(:).', ...
        'power_dbm', reportedPowerDbm(:).', ...
        'data', data, ...
        'selected_wavelength_nm', selectedWavelengthNm, ...
        'selected_power_dbm', selectedPowerDbm, ...
        'settings', settings ...
    );
end

function osa = localGetGpibObject(boardIndex, primaryAddress)
    existing = instrfind('Type', 'gpib', 'BoardIndex', boardIndex, ...
        'PrimaryAddress', primaryAddress, 'Tag', '');

    if isempty(existing)
        osa = gpib('NI', boardIndex, primaryAddress);
        return;
    end

    osa = existing(1);
    try
        fclose(osa);
    catch
    end
end

function localConfigureSweep(osa, settings)
    fprintf(osa, [':sens:wav:star ', num2str(settings.lam_start_nm), 'nm']);
    fprintf(osa, [':sens:wav:stop ', num2str(settings.lam_stop_nm), 'nm']);
    fprintf(osa, [':disp:trace:y1:rlev ', num2str(settings.reflevel_up_dbm), 'dbm']);
    fprintf(osa, [':sens:swe:points ', num2str(settings.point_count)]);
    fprintf(osa, [':sens:band:res ', num2str(settings.resolution_nm), 'nm']);
    fprintf(osa, [':sens:sens ', settings.sensitivity]);
    fprintf(osa, [':trac:attr:tr', settings.channel, ' write']);
    fprintf(osa, ':initiate:smode single');
    fprintf(osa, [':sens:swe:spe ', settings.speed]);
    fprintf(osa, [':trac:stat:tr', settings.channel, ' on']);
    fprintf(osa, '*CLS');
    fprintf(osa, ':init');
end

function localWaitForSweep(osa, timeoutSeconds)
    deadline = tic;
    while toc(deadline) < timeoutSeconds
        pause(0.1);
        fprintf(osa, ':stat:oper:even?');
        finish = str2double(strtrim(fscanf(osa)));
        if ~isnan(finish) && finish ~= 0
            return;
        end
    end

    error('wavelength_sweep:Timeout', 'Timed out while waiting for the OSA sweep to finish.');
end

function values = localParseAsciiTrace(rawText, expectedLength)
    tokens = textscan(char(rawText), '%f', 'Delimiter', ',');
    values = tokens{1}(:);

    if nargin > 1 && ~isempty(expectedLength) && numel(values) ~= expectedLength
        error('wavelength_sweep:UnexpectedPointCount', ...
            'Expected %d points from the OSA but received %d.', expectedLength, numel(values));
    end
end

function localCleanupOsa(osa, settings)
    if isempty(osa) || ~isvalid(osa)
        return;
    end

    if settings.restore_defaults
        try
            localRestoreDefaults(osa, settings);
        catch
        end
    end

    try
        if strcmpi(osa.Status, 'open')
            fclose(osa);
        end
    catch
    end

    try
        delete(osa);
    catch
    end
end

function localRestoreDefaults(osa, settings)
    if ~strcmpi(osa.Status, 'open')
        return;
    end

    if settings.use_requested_range
        fprintf(osa, [':sens:wav:star ', num2str(settings.lam_start_nm), 'nm']);
        fprintf(osa, [':sens:wav:stop ', num2str(settings.lam_stop_nm), 'nm']);
    else
        fprintf(osa, ':sens:wav:star 1520nm');
        fprintf(osa, ':sens:wav:stop 1610nm');
    end

    fprintf(osa, ':disp:trace:y1:rlev -10dbm');
    fprintf(osa, ':sens:band:res 0.1nm');
    fprintf(osa, ':sens:sens normal');
    fprintf(osa, ':sens:swe:points:auto on');
    fprintf(osa, ':sens:swe:spe 2x');
    fprintf(osa, ':initiate:smode repeat');
    fprintf(osa, ':init');
    pause(0.5);
end

function tf = localNumericScalar(value)
    tf = isnumeric(value) && isscalar(value) && isfinite(double(value));
end

function tf = localPositiveScalar(value)
    tf = localNumericScalar(value) && double(value) > 0;
end

function tf = localPositiveIntegerScalar(value)
    tf = localNumericScalar(value) && double(value) > 0 ...
        && mod(double(value), 1) == 0;
end

function tf = localNonNegativeIntegerScalar(value)
    tf = localNumericScalar(value) && double(value) >= 0 ...
        && mod(double(value), 1) == 0;
end

function tf = localLogicalScalar(value)
    tf = (islogical(value) || isnumeric(value)) && isscalar(value);
end

function tf = localTextScalar(value)
    tf = ischar(value) || (isstring(value) && isscalar(value));
end

function tf = localNumericVector(value)
    tf = isnumeric(value) && isvector(value);
end
