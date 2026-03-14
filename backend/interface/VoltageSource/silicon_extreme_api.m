function result = silicon_extreme_api(action, varargin)
%SILICON_EXTREME_API Persistent command interface for SiliconExtreme.
%   This wrapper is designed for MATLAB Engine callers that need a stable
%   function-based API instead of manipulating MATLAB handle objects.
%
%   Actions:
%       connect, COM_order
%       disconnect
%       status
%       set_voltage, channel, voltage
%       set_voltages, channels, voltages
%       configure_limits, channels, Vmax, Imax
%       read_voltage, channel
%       read_current, channel
%       read_power, channel
%       snapshot, channels

    persistent device activeComOrder

    if nargin < 1
        error('silicon_extreme_api:MissingAction', 'An action string is required.');
    end

    action = lower(char(action));

    switch action
        case {'connect', 'open'}
            localRequireArgumentCount(varargin, 1, action);
            comOrder = localScalar(varargin{1}, 'COM_order');
            [device, activeComOrder] = localConnect(device, activeComOrder, comOrder);
            result = struct( ...
                'connected', true, ...
                'com_order', activeComOrder ...
            );

        case {'disconnect', 'close'}
            localDisconnect(device);
            device = [];
            activeComOrder = [];
            result = struct('connected', false);

        case 'status'
            result = struct( ...
                'connected', ~isempty(device) && isvalid(device) && device.on_off == 1, ...
                'com_order', activeComOrder ...
            );

        otherwise
            if isempty(device) || ~isvalid(device)
                error('silicon_extreme_api:NotConnected', ...
                    'No active device. Call silicon_extreme_api(''connect'', COM_order) first.');
            end

            switch action
                case 'set_voltage'
                    localRequireArgumentCount(varargin, 2, action);
                    channel = localScalar(varargin{1}, 'channel');
                    voltage = localFiniteScalar(varargin{2}, 'voltage');
                    device.setV(channel, voltage);
                    result = struct('channel', channel, 'voltage', device.getV(channel));

                case 'set_voltages'
                    localRequireArgumentCount(varargin, 2, action);
                    channels = localVector(varargin{1}, 'channels');
                    voltages = localVector(varargin{2}, 'voltages');
                    if numel(channels) ~= numel(voltages)
                        error('silicon_extreme_api:InvalidInput', ...
                            'channels and voltages must have the same number of elements.');
                    end
                    device.setVoltages(channels, voltages);
                    result = device.snapshot(channels);

                case 'configure_limits'
                    localRequireArgumentCount(varargin, 3, action);
                    channels = localVector(varargin{1}, 'channels');
                    vmax = localOptionalVector(varargin{2}, 'Vmax');
                    imax = localOptionalVector(varargin{3}, 'Imax');
                    device.configureLimits(channels, vmax, imax);
                    result = struct( ...
                        'channels', reshape(channels, 1, []), ...
                        'vmax', reshape(vmax, 1, []), ...
                        'imax', reshape(imax, 1, []) ...
                    );

                case 'read_voltage'
                    localRequireArgumentCount(varargin, 1, action);
                    channel = localScalar(varargin{1}, 'channel');
                    result = device.getV(channel);

                case 'read_current'
                    localRequireArgumentCount(varargin, 1, action);
                    channel = localScalar(varargin{1}, 'channel');
                    result = device.getI(channel);

                case 'read_power'
                    localRequireArgumentCount(varargin, 1, action);
                    channel = localScalar(varargin{1}, 'channel');
                    result = device.getP(channel);

                case 'snapshot'
                    localRequireArgumentCount(varargin, 1, action);
                    channels = localVector(varargin{1}, 'channels');
                    result = device.snapshot(channels);

                otherwise
                    error('silicon_extreme_api:UnknownAction', ...
                        'Unsupported action: %s', action);
            end
    end
end

function [device, activeComOrder] = localConnect(device, activeComOrder, comOrder)
    if isempty(device) || ~isvalid(device) || isempty(activeComOrder) || activeComOrder ~= comOrder
        localDisconnect(device);
        device = SiliconExtreme(comOrder);
        activeComOrder = comOrder;
        return;
    end

    device.open();
end

function localDisconnect(device)
    if isempty(device) || ~isvalid(device)
        return;
    end

    try
        device.close();
    catch
    end

    try
        delete(device);
    catch
    end
end

function localRequireArgumentCount(argumentsIn, expectedCount, action)
    if numel(argumentsIn) < expectedCount
        error('silicon_extreme_api:MissingArgument', ...
            'Action %s requires at least %d arguments.', action, expectedCount);
    end
end

function value = localScalar(value, valueName)
    validateattributes(value, {'numeric'}, {'scalar', 'integer', 'positive'});
    value = double(value);
    if ~isfinite(value)
        error('silicon_extreme_api:InvalidValue', '%s must be finite.', valueName);
    end
end

function value = localFiniteScalar(value, valueName)
    validateattributes(value, {'numeric'}, {'scalar', 'finite'});
    value = double(value);
    if ~isfinite(value)
        error('silicon_extreme_api:InvalidValue', '%s must be finite.', valueName);
    end
end

function values = localVector(values, valueName)
    validateattributes(values, {'numeric'}, {'vector', 'finite'});
    values = reshape(double(values), 1, []);

    if isempty(values)
        error('silicon_extreme_api:InvalidValue', '%s cannot be empty.', valueName);
    end
end

function values = localOptionalVector(values, valueName)
    if isempty(values)
        values = [];
        return;
    end

    values = localVector(values, valueName);
end
