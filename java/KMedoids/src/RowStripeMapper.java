import java.io.Closeable;
import java.io.IOException;
import java.nio.*;
import java.nio.channels.FileChannel;
import java.nio.file.Path;

public final class RowStripeMapper implements Closeable {
	private static final long MAX_MAP_BYTES = Integer.MAX_VALUE; // ~2.147 GB
	
	private final FileChannel ch;
	private final int n;
	private final int rowsPerStripe;
	private final int stripeCount;
	
	private int curStripe = -1;
	private int curRowStart = -1;
	private int curRows = 0;
	private MappedByteBuffer mb;
	private FloatBuffer fb;
	
	public RowStripeMapper(Path path, int n) throws IOException {
		this.ch = FileChannel.open(path);
		this.n = n;
		
		// Each row has n floats -> n*4 bytes. Choose rowsPerStripe so rowsPerStripe*n*4 <= MAX_MAP_BYTES
		long bytesPerRow = (long) n * Float.BYTES;
		int rps = (int) Math.max(1, Math.min(n, MAX_MAP_BYTES / Math.max(1, bytesPerRow)));
		this.rowsPerStripe = rps;
		this.stripeCount = (n + rps - 1) / rps;
	}
	
	private void mapStripe(int stripe) throws IOException {
		if (stripe == curStripe) return;
		int rowStart = stripe * rowsPerStripe;
		int rows = Math.min(rowsPerStripe, n - rowStart);
		
		long offsetBytes  = (long) rowStart * n * Float.BYTES;
		long lengthBytes  = (long) rows * n * Float.BYTES;
		
		this.mb = ch.map(FileChannel.MapMode.READ_ONLY, offsetBytes, lengthBytes);
		this.fb = mb.order(ByteOrder.LITTLE_ENDIAN).asFloatBuffer();
		
		this.curStripe = stripe;
		this.curRowStart = rowStart;
		this.curRows = rows;
	}
	
	/** Get D[i,j] from row-major file. */
	public float get(int i, int j) throws IOException {
		if ((i | j) < 0 || i >= n || j >= n) throw new IndexOutOfBoundsException();
		int stripe = i / rowsPerStripe;
		mapStripe(stripe);
		int localRow = i - curRowStart;
		int idx = localRow * n + j;       // within the mapped stripe
		return fb.get(idx);
	}
	
	public int size() { return n; }
	
	@Override
	public void close() throws IOException { ch.close(); }
}
