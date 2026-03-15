function result = ramzi(config)
%RAMZI Standardized RAMZI photonic model entrypoint.
%   RESULT = RAMZI(CONFIG) accepts either a MATLAB struct or a JSON string.
%   The expected CONFIG layout is documented in backend/model/readme.md.

    config = localNormalizeConfig(config);
    resolved = localResolveConfig(config);

    wavelengthNm = resolved.simulation.wavelength_nm.start: ...
        resolved.simulation.wavelength_nm.step: ...
        resolved.simulation.wavelength_nm.stop;
    wavelengthNm = reshape(double(wavelengthNm), 1, []);

    if isempty(wavelengthNm)
        error('ramzi:InvalidWavelengthGrid', 'The wavelength grid is empty.');
    end

    fixed = resolved.parameters.fixed;
    tunable = resolved.parameters.tunable;

    lambda = wavelengthNm * 1e-9;
    frequencyHz = fixed.c ./ lambda;

    E1 = fixed.E1;
    E2 = fixed.E2;
    Ki = fixed.Ki;
    Ko = fixed.Ko;
    thetai = tunable.thetai;
    thetao = tunable.thetao;
    fait = tunable.fait;
    faib = tunable.faib;
    fai1 = tunable.fai1;
    fai2 = tunable.fai2;
    fai3 = tunable.fai3;
    fai4 = tunable.fai4;
    theta1 = fixed.theta1;
    theta2 = fixed.theta2;
    theta3 = fixed.theta3;
    theta4 = fixed.theta4;
    Alfadb = fixed.Alfadb;
    ng = fixed.ng;
    L1 = fixed.L1;
    L2 = fixed.L2;
    L3 = fixed.L3;
    L4 = fixed.L4;
    c = fixed.c;

    alfadb = Alfadb / 2;
    alfa = alfadb * log(10) / 10;

    tau1 = exp(-alfa * L1);
    tau2 = exp(-alfa * L2);
    tau3 = exp(-alfa * L3);
    tau4 = exp(-alfa * L4);

    phi1 = 2 * pi * frequencyHz * L1 * ng / c;
    phi2 = 2 * pi * frequencyHz * L2 * ng / c;
    phi3 = 2 * pi * frequencyHz * L3 * ng / c;
    phi4 = 2 * pi * frequencyHz * L4 * ng / c;

    E3 = sqrt(1 - Ki) * E1 - 1i * sqrt(Ki) * E2;
    E4 = -1i * sqrt(Ki) * E1 + sqrt(1 - Ki) * E2;
    E5 = E3;
    E6 = exp(1i * thetai) * E4;
    E7 = sqrt(1 - Ki) * E5 - 1i * sqrt(Ki) * E6;
    E8 = -1i * sqrt(Ki) * E5 + sqrt(1 - Ki) * E6;

    c1 = localRingBlock(theta1, fai1, phi1, tau1);
    c2 = localRingBlock(theta2, fai2, phi2, tau2);
    c3 = localRingBlock(theta3, fai3, phi3, tau3);
    c4 = localRingBlock(theta4, fai4, phi4, tau4);

    A1 = exp(1i * fait) .* c1 .* c2;
    A2 = exp(1i * faib) .* c3 .* c4;

    E9 = A1 .* E7;
    E10 = A2 .* E8;

    E11 = sqrt(1 - Ko) .* E9 - 1i * sqrt(Ko) .* E10;
    E12 = -1i * sqrt(Ko) .* E9 + sqrt(1 - Ko) .* E10;
    E13 = E11;
    E14 = exp(1i * thetao) .* E12;
    E15 = sqrt(1 - Ko) .* E13 - 1i * sqrt(Ko) .* E14;
    E16 = -1i * sqrt(Ko) .* E13 + sqrt(1 - Ko) .* E14;

    C1 = abs(E15) .^ 2;
    C2 = abs(E16) .^ 2;
    CdB1 = 10 * log10(max(C1, realmin('double')));
    CdB2 = 10 * log10(max(C2, realmin('double')));

    observePort = localCanonicalPortName(resolved.outputs.observe_port);
    switch observePort
        case 'C1'
            complexResponse = E15;
            powerLinear = C1;
            powerDb = CdB1;
        case 'C2'
            complexResponse = E16;
            powerLinear = C2;
            powerDb = CdB2;
        otherwise
            error('ramzi:UnsupportedPort', 'Unsupported observe_port: %s', observePort);
    end

    result = struct( ...
        'model_name', 'ramzi', ...
        'wavelength_nm', wavelengthNm, ...
        'frequency_hz', frequencyHz, ...
        'port_name', observePort, ...
        'complex_response', reshape(complexResponse, 1, []), ...
        'power_linear', reshape(powerLinear, 1, []), ...
        'power_db', reshape(powerDb, 1, []), ...
        'all_complex_response', struct('C1', reshape(E15, 1, []), 'C2', reshape(E16, 1, [])), ...
        'all_power_linear', struct('C1', reshape(C1, 1, []), 'C2', reshape(C2, 1, [])), ...
        'all_power_db', struct('C1', reshape(CdB1, 1, []), 'C2', reshape(CdB2, 1, [])), ...
        'parameters', resolved.parameters, ...
        'simulation', resolved.simulation ...
    );
end

function value = localRingBlock(theta, phiBias, phiRoundTrip, tau)
    value = ((exp(1i * theta) - 1) / 2 - tau .* exp(1i * (phiBias + theta + phiRoundTrip))) ...
        ./ (1 - tau .* (1 - exp(1i * theta)) .* exp(1i * (phiRoundTrip + phiBias)) / 2);
end

function config = localNormalizeConfig(config)
    if ischar(config) || (isstring(config) && isscalar(config))
        config = jsondecode(char(config));
    end

    if ~isstruct(config)
        error('ramzi:InvalidConfig', 'CONFIG must be a struct or a JSON string.');
    end
end

function resolved = localResolveConfig(config)
    defaults = localDefaultConfig();
    resolved = defaults;

    if isfield(config, 'simulation') && isfield(config.simulation, 'wavelength_nm')
        wavelengthConfig = config.simulation.wavelength_nm;
        resolved.simulation.wavelength_nm.start = localGetNumericField( ...
            wavelengthConfig, 'start', defaults.simulation.wavelength_nm.start);
        resolved.simulation.wavelength_nm.stop = localGetNumericField( ...
            wavelengthConfig, 'stop', defaults.simulation.wavelength_nm.stop);
        resolved.simulation.wavelength_nm.step = localGetNumericField( ...
            wavelengthConfig, 'step', defaults.simulation.wavelength_nm.step);
    end

    if resolved.simulation.wavelength_nm.stop <= resolved.simulation.wavelength_nm.start
        error('ramzi:InvalidWavelengthRange', ...
            'simulation.wavelength_nm.stop must be greater than start.');
    end

    if isfield(config, 'outputs') && isfield(config.outputs, 'observe_port')
        resolved.outputs.observe_port = localCanonicalPortName(config.outputs.observe_port);
    end

    if isfield(config, 'parameters') && isfield(config.parameters, 'tunable')
        [resolved.parameters.tunable, resolved.parameters.tunable_specs] = localResolveTunableParameterBlock( ...
            config.parameters.tunable, defaults.parameters.tunable, defaults.parameters.tunable_specs);
    end

    if isfield(config, 'parameters') && isfield(config.parameters, 'fixed')
        resolved.parameters.fixed = localResolveParameterBlock( ...
            config.parameters.fixed, defaults.parameters.fixed, false);
    end

    if isfield(config, 'parameters') && isfield(config.parameters, 'constraints')
        resolved.parameters.constraints = localNormalizeConstraintCollection(config.parameters.constraints);
    end
end

function defaults = localDefaultConfig()
    defaults = struct();
    defaults.simulation = struct( ...
        'wavelength_nm', struct( ...
            'start', 1549.9230, ...
            'stop', 1550.2032, ...
            'step', 0.0002 ...
        ) ...
    );
    defaults.outputs = struct('observe_port', 'C2');
    defaults.parameters = struct();
    defaults.parameters.tunable = struct( ...
        'thetai', 1.5707963267948966, ...
        'thetao', 1.5707963267948966, ...
        'fait', 1.5550883635269477, ...
        'faib', -1.5550883635269477, ...
        'fai1', -0.1470265361880023, ...
        'fai2', -2.1494776935861366, ...
        'fai3', -0.1627344994559513, ...
        'fai4', -1.9471591266949537 ...
    );
    defaults.parameters.fixed = struct( ...
        'E1', complex(0.0, 0.0), ...
        'E2', complex(1.0, 0.0), ...
        'Ki', 0.5, ...
        'Ko', 0.5, ...
        'theta1', -1.9540706305328512, ...
        'theta2', -2.2933626371205489, ...
        'theta3', -1.9540706305328512, ...
        'theta4', -2.2933626371205489, ...
        'Alfadb', 15.0, ...
        'ng', 4.3, ...
        'L1', 350e-6, ...
        'L2', 3000e-6, ...
        'L3', 350e-6, ...
        'L4', 3000e-6, ...
        'c', 3e8 ...
    );
    defaults.parameters.tunable_specs = localBuildDefaultTunableSpecs(defaults.parameters.tunable);
    defaults.parameters.constraints = {};
end

function specs = localBuildDefaultTunableSpecs(values)
    specs = struct();
    names = fieldnames(values);
    for idx = 1:numel(names)
        name = names{idx};
        specs.(name) = struct( ...
            'value', double(values.(name)), ...
            'bounds', [] ...
        );
    end
end

function [resolvedValues, resolvedSpecs] = localResolveTunableParameterBlock(inputBlock, defaults, defaultSpecs)
    if ~isstruct(inputBlock)
        error('ramzi:InvalidParameterBlock', 'Parameter blocks must be MATLAB structs.');
    end

    resolvedValues = defaults;
    resolvedSpecs = defaultSpecs;

    inputNames = fieldnames(inputBlock);
    for idx = 1:numel(inputNames)
        name = inputNames{idx};
        if ~isfield(defaults, name)
            error('ramzi:UnknownParameter', 'Unknown parameter: %s', name);
        end

        candidate = inputBlock.(name);
        bounds = defaultSpecs.(name).bounds;

        if isstruct(candidate) && isfield(candidate, 'value')
            resolvedValue = localResolveScalar(candidate.value);
            if isfield(candidate, 'bounds')
                bounds = localResolveBounds(candidate.bounds, name);
            end
        else
            resolvedValue = localResolveScalar(candidate);
        end

        resolvedValues.(name) = resolvedValue;
        resolvedSpecs.(name) = struct( ...
            'value', resolvedValue, ...
            'bounds', bounds ...
        );
    end
end

function resolved = localResolveParameterBlock(inputBlock, defaults, expectValueField)
    if ~isstruct(inputBlock)
        error('ramzi:InvalidParameterBlock', 'Parameter blocks must be MATLAB structs.');
    end

    resolved = defaults;
    inputNames = fieldnames(inputBlock);
    for idx = 1:numel(inputNames)
        name = inputNames{idx};
        if ~isfield(defaults, name)
            error('ramzi:UnknownParameter', 'Unknown parameter: %s', name);
        end

        candidate = inputBlock.(name);
        if expectValueField && isstruct(candidate) && isfield(candidate, 'value')
            resolved.(name) = localResolveScalar(candidate.value);
        else
            resolved.(name) = localResolveScalar(candidate);
        end
    end
end

function value = localResolveScalar(candidate)
    if isstruct(candidate) && isfield(candidate, 'real')
        imaginaryPart = 0.0;
        if isfield(candidate, 'imag')
            imaginaryPart = double(candidate.imag);
        end
        value = complex(double(candidate.real), imaginaryPart);
        return;
    end

    if ~(isnumeric(candidate) && isscalar(candidate))
        error('ramzi:InvalidScalar', 'Parameters must resolve to scalar numeric values.');
    end

    value = double(candidate);
end

function bounds = localResolveBounds(candidate, parameterName)
    if isempty(candidate)
        bounds = [];
        return;
    end

    if ~(isnumeric(candidate) && isvector(candidate) && numel(candidate) == 2)
        error('ramzi:InvalidBounds', 'Bounds for %s must be a numeric 2-element vector.', parameterName);
    end

    bounds = reshape(double(candidate), 1, []);
    if bounds(1) > bounds(2)
        error('ramzi:InvalidBounds', 'Bounds for %s must satisfy lower <= upper.', parameterName);
    end
end

function constraints = localNormalizeConstraintCollection(candidate)
    if isempty(candidate)
        constraints = {};
        return;
    end

    if iscell(candidate)
        constraints = reshape(candidate, 1, []);
        return;
    end

    if isstruct(candidate)
        constraints = cell(1, numel(candidate));
        for idx = 1:numel(candidate)
            constraints{idx} = candidate(idx);
        end
        return;
    end

    error('ramzi:InvalidConstraints', 'parameters.constraints must be a struct array or a cell array.');
end

function value = localGetNumericField(source, fieldName, defaultValue)
    if isfield(source, fieldName)
        value = double(source.(fieldName));
    else
        value = double(defaultValue);
    end
end

function portName = localCanonicalPortName(portValue)
    portName = upper(char(string(portValue)));
    switch portName
        case 'BAR'
            portName = 'C1';
        case 'CROSS'
            portName = 'C2';
    end

    if ~strcmp(portName, 'C1') && ~strcmp(portName, 'C2')
        error('ramzi:InvalidPort', 'observe_port must be C1, C2, BAR, or CROSS.');
    end
end
