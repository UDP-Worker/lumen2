classdef SiliconExtreme < handle

    properties
        port
        on_off
        com_order
    end

    methods
        function obj = SiliconExtreme(COM_order, auto_open)
            if nargin < 2
                auto_open = true;
            end

            validateattributes(COM_order, {'numeric'}, {'scalar', 'integer', 'positive'});
            localReleaseExistingPort(COM_order);

            obj.com_order = double(COM_order);
            obj.port = serial(sprintf('COM%d', COM_order), 'baudrate', 115200, ...
                'parity', 'none', 'databits', 8, 'stopbits', 1);
            obj.on_off = 0;

            if auto_open
                obj.open();
            end
        end

        function delete(obj)
            if isempty(obj.port) || ~isvalid(obj.port)
                return;
            end

            try
                obj.close();
            catch
            end

            try
                delete(obj.port);
            catch
            end
        end

        function IsSuccess = open(obj)
            if obj.on_off == 1 && strcmpi(obj.port.Status, 'open')
                IsSuccess = true;
                return;
            end

            fopen(obj.port);
            obj.on_off = 1;
            IsSuccess = true;
        end

        function IsSuccess = close(obj)
            if isempty(obj.port) || ~isvalid(obj.port)
                obj.on_off = 0;
                IsSuccess = true;
                return;
            end

            if strcmpi(obj.port.Status, 'open')
                fclose(obj.port);
            end

            obj.on_off = 0;
            IsSuccess = true;
        end

        function answer = query(obj, text)
            obj.open();
            fprintf(obj.port, text);
            answer = strtrim(fscanf(obj.port));
        end

        function V = getV(obj, channel)
            V = obj.readNumeric(sprintf('V%d?', obj.validateChannel(channel)), 'voltage');
        end

        function IsSuccess = setV(obj, channel, V)
            validateattributes(V, {'numeric'}, {'scalar', 'finite'});
            IsSuccess = obj.expectOk(sprintf('V%d=%.4f', obj.validateChannel(channel), V), ...
                'set voltage');
        end

        function I = getI(obj, channel)
            I = obj.readNumeric(sprintf('I%d?', obj.validateChannel(channel)), 'current');
        end

        function IsSuccess = setI(obj, channel, I)
            validateattributes(I, {'numeric'}, {'scalar', 'finite'});
            IsSuccess = obj.expectOk(sprintf('I%d=%.4f', obj.validateChannel(channel), I), ...
                'set current');
        end

        function P = getP(obj, channel)
            P = obj.readNumeric(sprintf('P%d?', obj.validateChannel(channel)), 'power');
        end

        function IsSuccess = setVmax(obj, channel, Vmax)
            validateattributes(Vmax, {'numeric'}, {'scalar', 'finite'});
            IsSuccess = obj.expectOk(sprintf('VMAX%d=%.4f', obj.validateChannel(channel), Vmax), ...
                'set VMAX');
        end

        function IsSuccess = setImax(obj, channel, Imax)
            validateattributes(Imax, {'numeric'}, {'scalar', 'finite'});
            IsSuccess = obj.expectOk(sprintf('IMAX%d=%.4f', obj.validateChannel(channel), Imax), ...
                'set IMAX');
        end

        function IsSuccess = setVmax_all(obj, Vmax)
            validateattributes(Vmax, {'numeric'}, {'scalar', 'finite'});
            IsSuccess = obj.expectOk(sprintf('VMAXALL=%.4f', Vmax), 'set VMAXALL');
        end

        function IsSuccess = setImax_all(obj, Imax)
            validateattributes(Imax, {'numeric'}, {'scalar', 'finite'});
            IsSuccess = obj.expectOk(sprintf('IMAXALL=%.4f', Imax), 'set IMAXALL');
        end

        function setVoltages(obj, channels, voltages)
            channels = obj.validateChannelVector(channels);
            voltages = obj.validateNumericVector(voltages, numel(channels), 'voltages');

            for idx = 1:numel(channels)
                obj.setV(channels(idx), voltages(idx));
            end
        end

        function configureLimits(obj, channels, vmax, imax)
            channels = obj.validateChannelVector(channels);
            vmaxValues = obj.expandScalarOrVector(vmax, numel(channels), 'Vmax');
            imaxValues = obj.expandScalarOrVector(imax, numel(channels), 'Imax');

            if ~isempty(vmaxValues)
                for idx = 1:numel(channels)
                    obj.setVmax(channels(idx), vmaxValues(idx));
                end
            end

            if ~isempty(imaxValues)
                for idx = 1:numel(channels)
                    obj.setImax(channels(idx), imaxValues(idx));
                end
            end
        end

        function snapshot = snapshot(obj, channels)
            channels = obj.validateChannelVector(channels);

            voltages = zeros(size(channels));
            currents = zeros(size(channels));
            powers = zeros(size(channels));

            for idx = 1:numel(channels)
                voltages(idx) = obj.getV(channels(idx));
                currents(idx) = obj.getI(channels(idx));
                try
                    powers(idx) = obj.getP(channels(idx));
                catch
                    % Some firmware revisions do not implement per-channel power reads.
                    powers(idx) = NaN;
                end
            end

            snapshot = struct( ...
                'channels', reshape(channels, 1, []), ...
                'voltages', reshape(voltages, 1, []), ...
                'currents', reshape(currents, 1, []), ...
                'powers', reshape(powers, 1, []) ...
            );
        end
    end

    methods (Access = private)
        function value = readNumeric(obj, commandText, quantityName)
            response = obj.query(commandText);
            value = sscanf(response, '%f', 1);

            if isempty(value) || isnan(value)
                error('SiliconExtreme:InvalidResponse', ...
                    'Failed to read %s. Raw response: %s', quantityName, response);
            end
        end

        function IsSuccess = expectOk(obj, commandText, actionName)
            response = obj.query(commandText);
            IsSuccess = strcmpi(strtrim(response), 'OK');

            if ~IsSuccess
                error('SiliconExtreme:CommandFailed', ...
                    'Failed to %s. Raw response: %s', actionName, response);
            end
        end

        function channel = validateChannel(~, channel)
            validateattributes(channel, {'numeric'}, {'scalar', 'integer', 'positive'});
            channel = double(channel);
        end

        function channels = validateChannelVector(obj, channels)
            validateattributes(channels, {'numeric'}, {'vector', 'integer', 'positive'});
            channels = reshape(double(channels), 1, []);

            for idx = 1:numel(channels)
                obj.validateChannel(channels(idx));
            end
        end

        function values = validateNumericVector(~, values, expectedLength, valueName)
            validateattributes(values, {'numeric'}, {'vector', 'numel', expectedLength, 'finite'});
            values = reshape(double(values), 1, []);

            if numel(values) ~= expectedLength
                error('SiliconExtreme:InvalidVectorLength', ...
                    '%s must contain %d elements.', valueName, expectedLength);
            end
        end

        function values = expandScalarOrVector(obj, values, expectedLength, valueName)
            if isempty(values)
                values = [];
                return;
            end

            validateattributes(values, {'numeric'}, {'vector', 'finite'});
            values = reshape(double(values), 1, []);

            if isscalar(values)
                values = repmat(values, 1, expectedLength);
            end

            values = obj.validateNumericVector(values, expectedLength, valueName);
        end
    end
end

function localReleaseExistingPort(COM_order)
    existingPorts = instrfind('Port', sprintf('COM%d', COM_order));
    for idx = 1:numel(existingPorts)
        try
            fclose(existingPorts(idx));
        catch
        end

        try
            delete(existingPorts(idx));
        catch
        end
    end
end
