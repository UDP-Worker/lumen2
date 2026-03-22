function result = SWEEP_SiliconExtreme(COM_order, channels, voltages, varargin)
%SWEEP_SILICONEXTREME Apply voltages to multiple SiliconExtreme channels.
%   RESULT = SWEEP_SILICONEXTREME(COM_order, channels, voltages, Name, Value, ...)
%   connects to the device, optionally configures current/voltage limits,
%   writes the requested voltages, waits for the specified settle time, and
%   returns the measured snapshot for the updated channels.
%
%   Name-value parameters:
%       Vmax                 empty | scalar | vector, default []
%       Imax                 empty | scalar | vector, default []
%       settle_time_s        scalar >= 0, default 0.5
%       disconnect_when_done logical/numeric scalar, default true

    parser = inputParser;
    parser.FunctionName = 'SWEEP_SiliconExtreme';
    addParameter(parser, 'Vmax', [], @localEmptyOrNumericVector);
    addParameter(parser, 'Imax', [], @localEmptyOrNumericVector);
    addParameter(parser, 'settle_time_s', 0.5, @localNonNegativeScalar);
    addParameter(parser, 'disconnect_when_done', true, @localLogicalScalar);
    parse(parser, varargin{:});
    options = parser.Results;

    validateattributes(COM_order, {'numeric'}, {'scalar', 'integer', 'positive'});
    validateattributes(channels, {'numeric'}, {'vector', 'integer', 'positive'});
    validateattributes(voltages, {'numeric'}, {'vector', 'finite'});

    channels = reshape(double(channels), 1, []);
    voltages = reshape(double(voltages), 1, []);

    if numel(channels) ~= numel(voltages)
        error('SWEEP_SiliconExtreme:InvalidInput', ...
            'channels and voltages must have the same number of elements.');
    end

    silicon_extreme_api('connect', COM_order);
    cleanup = []; %#ok<NASGU>
    if options.disconnect_when_done
        cleanup = onCleanup(@() silicon_extreme_api('disconnect'));
    end

    if ~isempty(options.Vmax) || ~isempty(options.Imax)
        silicon_extreme_api('configure_limits', channels, options.Vmax, options.Imax);
    end

    silicon_extreme_api('set_voltages', channels, voltages);

    if options.settle_time_s > 0
        pause(options.settle_time_s);
    end

    result = silicon_extreme_api('snapshot', channels);
    result.com_order = double(COM_order);
    result.requested_voltages = voltages;
end

function tf = localEmptyOrNumericVector(value)
    tf = isempty(value) || isnumeric(value);
end

function tf = localLogicalScalar(value)
    tf = (islogical(value) || isnumeric(value)) && isscalar(value);
end

function tf = localNonNegativeScalar(value)
    tf = isnumeric(value) && isscalar(value) && isfinite(double(value)) ...
        && double(value) >= 0;
end
