import socket
import struct
import time

# =========================================================
# ATI NETrs / Gamma UDP RDT reader
# =========================================================

SENSOR_IP = "192.168.1.1"
SENSOR_PORT = 49152

# Put your calibration scaling here if you know them.
# If you do not know them yet, leave as None and the script
# will print raw counts.
COUNTS_PER_FORCE = 1000000  # example: 1000000.0
COUNTS_PER_TORQUE = 1000000  # example: 1000000.0

# RDT commands from ATI manual
RDT_HEADER = 0x1234
CMD_STOP = 0x0000
CMD_START_REALTIME = 0x0002

# One RDT record = 3 uint32 + 6 int32 = 36 bytes
RDT_RECORD_FMT = "!IIIiiiiii"
RDT_RECORD_SIZE = struct.calcsize(RDT_RECORD_FMT)


def build_rdt_request(command: int, sample_count: int = 0) -> bytes:
    """
    Build ATI RDT request packet.
    sample_count = 0 means continuous streaming until STOP is sent.
    """
    return struct.pack("!HHI", RDT_HEADER, command, sample_count)


def counts_to_units(fx, fy, fz, tx, ty, tz):
    """
    Convert counts to engineering units if scaling is provided.
    Otherwise return raw counts.
    """
    if COUNTS_PER_FORCE is None or COUNTS_PER_TORQUE is None:
        return {
            "Fx": fx, "Fy": fy, "Fz": fz,
            "Tx": tx, "Ty": ty, "Tz": tz,
            "units": "counts"
        }

    return {
        "Fx": fx / COUNTS_PER_FORCE,
        "Fy": fy / COUNTS_PER_FORCE,
        "Fz": fz / COUNTS_PER_FORCE,
        "Tx": tx / COUNTS_PER_TORQUE,
        "Ty": ty / COUNTS_PER_TORQUE,
        "Tz": tz / COUNTS_PER_TORQUE,
        "units": "scaled"
    }


def main():
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.settimeout(2.0)

    try:
        # Bind to any available local UDP port
        sock.bind(("", 0))
        local_ip, local_port = sock.getsockname()
        print(f"Local UDP socket bound to {local_ip}:{local_port}")

        # Start continuous real-time streaming
        start_packet = build_rdt_request(CMD_START_REALTIME, sample_count=0)
        sock.sendto(start_packet, (SENSOR_IP, SENSOR_PORT))
        print(f"Started streaming from {SENSOR_IP}:{SENSOR_PORT}")
        print("Press Ctrl+C to stop.\n")

        last_rdt_seq = None

        while True:
            data, addr = sock.recvfrom(4096)

            if len(data) < RDT_RECORD_SIZE:
                print(f"Received short packet ({len(data)} bytes), skipping.")
                continue

            # Real-time mode usually gives one 36-byte record per UDP packet.
            # But in case multiple records arrive, parse them all.
            num_records = len(data) // RDT_RECORD_SIZE

            for i in range(num_records):
                chunk = data[i * RDT_RECORD_SIZE:(i + 1) * RDT_RECORD_SIZE]
                rdt_seq, ft_seq, status, fx, fy, fz, tx, ty, tz = struct.unpack(
                    RDT_RECORD_FMT, chunk
                )

                # Detect dropped UDP records
                dropped = ""
                if last_rdt_seq is not None and ((last_rdt_seq + 1) & 0xFFFFFFFF) != rdt_seq:
                    dropped = f"  [WARNING: expected {((last_rdt_seq + 1) & 0xFFFFFFFF)}, got {rdt_seq}]"
                last_rdt_seq = rdt_seq

                vals = counts_to_units(fx, fy, fz, tx, ty, tz)

                if vals["units"] == "counts":
                    print(
                        f"rdt_seq={rdt_seq:10d}  ft_seq={ft_seq:10d}  status={status:10d}  "
                        f"Fx={vals['Fx']:10d}  Fy={vals['Fy']:10d}  Fz={vals['Fz']:10d}  "
                        f"Tx={vals['Tx']:10d}  Ty={vals['Ty']:10d}  Tz={vals['Tz']:10d}{dropped}"
                    )
                else:
                    print(
                        f"rdt_seq={rdt_seq:10d}  ft_seq={ft_seq:10d}  status={status:10d}  "
                        f"Fx={vals['Fx']: .6f}  Fy={vals['Fy']: .6f}  Fz={vals['Fz']: .6f}  "
                        f"Tx={vals['Tx']: .6f}  Ty={vals['Ty']: .6f}  Tz={vals['Tz']: .6f}{dropped}"
                    )

    except KeyboardInterrupt:
        print("\nStopping stream...")

    finally:
        try:
            stop_packet = build_rdt_request(CMD_STOP, sample_count=0)
            sock.sendto(stop_packet, (SENSOR_IP, SENSOR_PORT))
            time.sleep(0.1)
        except Exception:
            pass

        sock.close()
        print("Socket closed.")


if __name__ == "__main__":
    main()
